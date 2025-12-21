#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensors_tools_msgs.srv import SetQueries

import torch
import gc

from sensors_tools.inference.models.naradio.naradio import NARadioEncoder


class QueryServer(Node):
    """
    Reads comma-separated text queries, lazily loads Naradio encoder,
    computes embeddings, sends them via ROS service, then unloads the model.
    """

    def __init__(self):
        super().__init__('query_server')

        self.cli = self.create_client(SetQueries, '/bloomxai_server/set_queries')

        while not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /bloomxai_server/set_queries service...')

        self.encoder = None
        self.device = None

        self.get_logger().info("QueryServer ready. Type comma-separated queries:")

        self.run_input_loop()

    def lazy_load_model(self):
        """Load Naradio model only when needed."""
        if self.encoder is not None:
            return
        
        self.get_logger().info("Loading Naradio text encoder (lazy load)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = NARadioEncoder(
            model_version="radio_v2.5-b",
            lang_model="siglip",
            device=self.device,
            input_resolution=(720, 720),
            compile=False
        )  
        self.get_logger().info("Model loaded successfully.")

    def unload_model(self):
        """Unload model to free GPU memory."""
        if self.encoder is None:
            return

        self.get_logger().info("Unloading Naradio model to free GPU memory...")

        # Explicitly delete model
        del self.encoder
        self.encoder = None

        # Force garbage collection
        gc.collect()

        # Free CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        self.get_logger().info("Model unloaded.")

    def run_input_loop(self):
        """Waits for user input."""
        try:
            while rclpy.ok():
                line = input("\nEnter queries (comma-separated) e.g., floor,cables,pebbles,animal,robot,cardboard: ").strip()
                if not line:
                    continue

                queries = [q.strip() for q in line.split(",") if q.strip()]
                self.process_queries(queries)

        except KeyboardInterrupt:
            pass

    def process_queries(self, queries):
        # Load model on demand
        self.lazy_load_model()

        self.get_logger().info(f"Encoding {len(queries)} queries...")

        # Encode
        with torch.no_grad():
            print(f"Input to encode: {queries}")
            text_features = self.encoder.encode_labels(queries)
            text_features = text_features.cpu().numpy()

        # Prepare service request
        num_queries = text_features.shape[0]
        sem_dim = text_features.shape[1]
        flat = text_features.flatten().tolist()

        req = SetQueries.Request()
        req.num_queries = num_queries
        req.flat_embeddings = flat

        self.get_logger().info(
            f"Sending {num_queries} queries with embedding dimension {sem_dim}"
        )

        # Call service
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            resp = future.result()
            self.get_logger().info(f"Server response: {resp.success}, {resp.message}")
        else:
            self.get_logger().error("Service call failed.")

        # Fully unload model to free GPU memory
        self.unload_model()

def main(args=None):
    rclpy.init(args=args)
    node = QueryServer()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
