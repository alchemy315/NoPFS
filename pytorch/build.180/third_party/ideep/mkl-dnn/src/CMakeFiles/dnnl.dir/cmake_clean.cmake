file(REMOVE_RECURSE
  "../../../../lib/libdnnl.pdb"
  "../../../../lib/libdnnl.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang C CXX)
  include(CMakeFiles/dnnl.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
