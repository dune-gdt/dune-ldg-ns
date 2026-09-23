add_executable(ldg_ns_dfg_cylinder dfg_cylinder.cc)
target_link_libraries(ldg_ns_dfg_cylinder PRIVATE ${DUNE_LIBS})
dune_symlink_to_source_files(FILES dfg_2d_2.ini dfg_2d_3.ini)
dune_symlink_to_source_files(FILES ../grids/dfg-cylinder-2d.geo)
