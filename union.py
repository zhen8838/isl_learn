import islpy as isl
# s1 = isl.UnionSet("{ B[0]; A[2 ,8 ,1] }")
# s2 = isl.UnionSet("{ A[2 ,8 ,1]; C[5] }")
s1 = isl.UnionMap("{ B[0] -> A[2 ,8 ,1] }")

s2 = isl.UnionMap("{ B[0] -> B[5] }")

s1.intersect(s2)

s1.reverse()

s2 = isl.UnionMap("{ B[0] -> B[5]; B[5] -> B[6]}")
s2.reverse()

A = isl.UnionMap("{ B[6] -> A[2 ,8 ,1]; B[6] -> B[5] }")
B = isl.UnionMap("{ A[2 ,8 ,1] -> B[5]; A[2 ,8 ,1] -> B[6]; B[5] -> B[5] }")

A.apply_range(B)
B.apply_range(A)

A.fixed_power_val(-1)
A.apply_range(A)
A.reverse().apply_range(A.reverse())  # R-2

R = isl.UnionMap("{ A[2 ,8 ,1] -> B[5]; A[2 ,8 ,1] -> B[6]; B[5] -> B[5] }")
R
R.fixed_power_val(2)  # R^2
R.apply_range(R)  # R^2
R.fixed_power_val(3) == R.apply_range(R).apply_range(R)  # true


R = isl.UnionMap("{ A[2 ,8 ,1] -> B[5]; A[2 ,8 ,1] -> B[6]; B[5] -> B[5] }")
R.range() # UnionSet("{ B[6]; B[5] }")
R.domain() # UnionSet("{ B[5]; A[2, 8, 1] }")


isl.UnionMapList( '[n] -> { A[i] : i >= 0 }')
# ; B := [n] -> { A[i] : i >= 0 and n >= 0 }; A = B;