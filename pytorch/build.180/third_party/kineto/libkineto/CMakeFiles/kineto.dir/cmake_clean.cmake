file(REMOVE_RECURSE
  "../../../lib/libkineto.pdb"
  "../../../lib/libkineto.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/kineto.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
