  // for (i = 0; i < partition->num_allocations; i++) {
  //   for (j = 0; j < partition->num_blocks; j++) {
  //     if (partition->allocations[i].num_target_blocks > 0) {
  //       for (k = 0; k < partition->allocations[i].num_target_blocks; k++) {
  //         if (partition->allocations[i].target_block_ids[k] == j) {
  //           target_block_ids[i][k] = j;
  //           used_block_ids[u] = j;
  //           u++;
  //           break;
  //         }
  //       }
  //     }
  //   }
  // }

  // bool found = false;

  // for (i = 0; i < partition->num_allocations; i++) {
  //   if (partition->allocations[i].num_target_blocks == 0) {
  //     for (j = 0; j < partition->num_blocks; j++) {
  //       for (k = 0; k < descs[i].num_blocks; k++) {
  //         for (l = 0; l < u; l++) {
  //           if (used_block_ids[l] == j) {
  //             found = true;
  //             break;
  //           }
  //         }
  //         if (!found) {
  //           target_block_ids[i][k] = j;
  //           found = false;
  //           break;
  //         }
  //       }
  //     }
  //   }
  // }
