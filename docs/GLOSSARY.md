# Glossary

| term | meaning in this project |
| --- | --- |
| **tier / level** | `OptLevel` 0-3: 0 reference, 1 production-proven ports, 2 engine features, 3 approximations |
| **resident layer** | a decoder layer whose weights live on the GPU permanently (4-5 on this box) |
| **streamed layer** | a layer copied from pinned host memory into a ring buffer every token |
| **ring** | the N device buffers the streamed layers rotate through |
| **slab** | one pinned host block per streamed layer, laid out exactly like the device buffer |
| **slot** | a VRAM buffer holding one routed expert's fp4 blob |
| **RAM record / slot record** | an `ExpertRecord`: the blob either in the RAM cache or in a VRAM slot |
| **expert cache** | the ARC/LRU policy over 18.8MB expert blobs on the host (14-15GB here) |
| **MLA** | multi-head latent attention, the model's attention family |
| **window layer** | layer with `compress_ratio == 0`: plain sliding-window attention |
| **compressed layer** | layer that pools KV into a latent and selects slots with an indexer |
| **compress_ratio** | the pooling factor (2 on this checkpoint) |
| **indexer** | the top-k slot selector used by compressed layers |
| **candidate blocks** | the coarse block selection a kv_source layer publishes for its followers |
| **kv_source / index_source** | the layer ids that publish shared compressed state |
| **hc_mult** | hyper-connection expansion factor (4) |
| **engram** | the n-gram hash table layers (1, 14) |
| **MTP / DSpark** | multi-token-prediction draft stages (3 here) in the checkpoint |
| **speculative decoding** | verify k drafts with one main forward, accept the matching prefix |
| **IO floor** | the bytes every token must move divided by the measured rates |
| **DeepInfer** | DeepSeek's serving stack, the semantic authority for this engine |
| **b12x** | NVIDIA's SM120 NVFP4 kernel library, used as algorithm reference only |
| **Flare** | NVIDIA's streaming/offload design, source of several level-1 mechanisms |
| **parity** | max relative difference between this engine and the reference for a component |
