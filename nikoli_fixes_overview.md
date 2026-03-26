## Summary

  This PR applies most of Nikoli's feedback and adds additional updates. Improves training-step behavior, data loading performance, dataset format efficiency, and warmup/profiling visibility.                                                  

  ## Changes

  - Switched optimizer updates from once per epoch to once per batch in the training loop.                                                                                        
  - Corrected AMP gradient handling so gradients are unscaled before clip_grad_norm_.
  - Extracted training lifecycle phases so prepare_training, warmup, and main train run as separate regions in traces.                                                            
  - Changed warmup from epoch-based to batch-based with new warmup_batches support, defaulting to 5 batches per rank.                                                             
  - Updated warmup to better match real training:
      - runs in model.train()
      - follows the same DDP + DistConv tensor distribution path as the main training loop                                                                                        
      - performs backward passes without optimizer steps

  ## Data Pipeline

  - Added configurable dataloader_num_workers in config and CLI.
  - Enabled persistent_workers and prefetch_factor when worker processes are used.
  - Optimized generated dataset format:
      - images now save in final float32 CDHW layout
      - masks now save in final int64 training dtype
  - Added dual-format dataset loading:
      - fast path for new optimized datasets
      - fallback path for legacy datasets that still need transpose/remap work

  ## Dataset Reuse

  - Added dataset_format_version metadata for generated datasets.
  - Included dataset_format_version in dataset reuse validation and hashed dataset identity so old/new dataset formats do not share the same cache key.                           

  ## Config / CLI Additions

  - dataloader_num_workers
  - warmup_batches