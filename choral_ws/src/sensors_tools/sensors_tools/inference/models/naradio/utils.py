"""Utility functions supporting the other modules"""
import torch

def compute_cos_sim(vec1: torch.FloatTensor,
                    vec2: torch.FloatTensor,
                    softmax: bool = False) -> torch.FloatTensor:
  """Compute cosine similarity between two batches of D dim vectors.
  
  Args:
    vec1: NxC float tensor representing batch of vectors
    vec2: MxC float tensor representing batch of vectors
    softmax: If False, cosine similarity is returned. If True, softmaxed
      probability is returned across the N dimension.
  Returns:
    result: MxN float tensor representing similarity/prob. where result[0,1]
      represents the similarity of vec1[0] with vec2[1]
  """
  N, C1 = vec1.shape
  M, C2 = vec2.shape
  if C1 != C2:
    raise ValueError(f"vec1 feature dimension '{C1}' does not match vec2"
                     f"feature dimension '{C2}'")
  C = C1

  vec1 = vec1 / vec1.norm(dim = -1, keepdim = True)
  vec1 = vec1.reshape(1, N, 1, C)

  vec2 = vec2 / vec2.norm(dim=-1, keepdim = True)
  vec2 = vec2.reshape(M, 1, C, 1)

  sim = (vec1 @ vec2).reshape(M, N)
  if softmax:
    return torch.softmax(100 * sim, dim = -1)
  else:
    return sim