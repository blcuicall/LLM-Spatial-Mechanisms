# Algorithm: Multi-hop Spatial Reasoning Dataset Generation

## Core Algorithm

```
Algorithm: GenerateSpatialReasoningDataset(N, k_min, k_max)
Input: N - number of samples, k_min, k_max - reasoning steps range
Output: D - dataset of spatial reasoning problems

1:  D ← ∅
2:  for i ← 1 to N do
3:      k ← Random(k_min, k_max)
4:      
5:      // Generate relation chain: E_0 → E_1 → ... → E_k
6:      E ← {A, B, C, ...} (k+1 entities)
7:      P[E_0] ← (0, 0, 0)
8:      R ← ∅
9:      
10:     for j ← 1 to k do
11:         E_ref ← Random(E_0, ..., E_{j-1})
12:         r ← Random({left, right, front, behind, above, below})
13:         P[E_j] ← P[E_ref] + V(r)
14:         R.append((E_j, r, E_ref))
15:     end for
16:     
17:     // Verify multi-hop requirement
18:     if ∃r ∈ R: (r.subject, r.object) ∈ {(E_0, E_k), (E_k, E_0)} then
19:         continue  // Skip if direct relation exists
20:     end if
21:     
22:     // Generate question and answer
23:     v ← P[E_k] - P[E_0]  // Relative position vector
24:     a ← VectorToRelation(v)
25:     Q ← ConstructQuestion(R, E_0, E_k)
26:     
27:     D.append({question: Q, answer: a, target: v, hops: k})
28: end for
29: return D

where V(r) = {
    left: (-1,0,0), right: (1,0,0),
    front: (0,1,0), behind: (0,-1,0),
    above: (0,0,1), below: (0,0,-1)
}
```

---

## Key Components

**Vector-to-Relation Mapping:**
```
VectorToRelation(v = (x, y, z)):
    return [
        "above" if z > 0 else "below" if z < 0 else "",
        "front" if y > 0 else "behind" if y < 0 else "",
        "right" if x > 0 else "left" if x < 0 else ""
    ].join(" and ")
```

**Properties:**
- **3D Coordinate System**: (x, y, z) = (left/right, behind/front, below/above)
- **Multi-hop Guarantee**: No direct relation between E₀ and Eₖ
- **Transitivity**: Answer computed via vector sum: v = Σᵢ V(rᵢ)
- **Complexity**: k-hop reasoning requires k transitive inferences

