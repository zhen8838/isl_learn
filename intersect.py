import islpy as isl
s1 = isl.UnionSet("{ B[0]; A[2 ,8 ,1] }")
s2 = isl.UnionSet("{ A[2 ,8 ,1]; C[5] }")
print(s1.intersect(s2))  # { A[2, 8, 1] }

s1 = isl.UnionSet("{ []; A[2 ,8 ,1] }")
s2 = isl.UnionSet("{ A[2 ,8 ,1]; []; [] }")
print(s1.is_equal(s2))  # { A[2, 8, 1] }
print(s1.is_subset(s2)) # A \ B = ∅
print(s1.is_strict_subset(s2)) # A \ B = ∅ and A!= B.
