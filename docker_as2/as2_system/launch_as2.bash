#!/bin/bash

usage() {
    echo "  options:"
    echo "      -n: select drone namespace to launch. Default is 'drone0'"
    echo "      -c: motion controller plugin (pid_speed_controller, differential_flatness_controller), choices: [pid, df]. Default: pid"
    echo "      -r: record rosbag. Default not launch"
}

# Initialize variables with default values
drone_namespace=""
motion_controller_plugin="pid"
rosbag="false"

# Arg parser
while getopts "n:r" opt; do
  case ${opt} in
    n )
      drone_namespace="${OPTARG}"
      ;;
    r )
      rosbag="true"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    : )
      if [[ ! $OPTARG =~ ^[wrt]$ ]]; then
        echo "Option -$OPTARG requires an argument" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

# If no drone namespaces are provided, finish the execution
if [ -z "$drone_namespace" ]; then
  echo "No drone namespace provided. Set it using the -n option"
  exit 1
fi

# Check if motion controller plugins are valid
case ${motion_controller_plugin} in
  pid )
    motion_controller_plugin="pid_speed_controller"
    ;;
  df )
    motion_controller_plugin="differential_flatness_controller"
    ;;
  * )
    echo "Invalid motion controller plugin: ${motion_controller_plugin}" >&2
    usage
    exit 1
    ;;
esac

# Launch aerostack2 for each drone namespace
eval "tmuxinator start -n ${drone_namespace} -p tmuxinator/aerostack2.yaml \
    drone_namespace=${drone_namespace} \
    motion_controller_plugin=${motion_controller_plugin} \
    rosbag=${rosbag}"
    wait

# Attach to tmux session
tmux attach-session -t ${drone_namespace}
