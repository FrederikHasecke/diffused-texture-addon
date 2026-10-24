# Changelog

All notable changes to this project will be documented in this file.

## [0.3.4](https://github.com/FrederikHasecke/diffused-texture-addon/compare/diffused_texture_addon-v0.3.3...diffused_texture_addon-v0.3.4) (2026-10-24)


### Bug Fixes

* **deps:** limit setuptools floor to the Python 3.13 runtime lane ([9d6cfaf](https://github.com/FrederikHasecke/diffused-texture-addon/commit/9d6cfaf54526287e5148cdec71a3edbf01bbe471))
* **deps:** update vulnerable runtime dependencies ([790a2c3](https://github.com/FrederikHasecke/diffused-texture-addon/commit/790a2c3e18bcb0c7e5865e0f2ab2c6457b833f6e))
* **deps:** update vulnerable runtime dependencies ([e3e9b79](https://github.com/FrederikHasecke/diffused-texture-addon/commit/e3e9b7985a7b40556c98a61a14c711ebeff63693))

## [0.3.3](https://github.com/FrederikHasecke/diffused-texture-addon/compare/diffused_texture_addon-v0.3.2...diffused_texture_addon-v0.3.3) (2026-10-11)


### Bug Fixes

* **bake:** select the target object before baking and restore selection ([5ea2ec4](https://github.com/FrederikHasecke/diffused-texture-addon/commit/5ea2ec4d7fd2696fca36734b5b1beb0d866f381e))
* **bake:** select the target object before baking and restore selection ([2215621](https://github.com/FrederikHasecke/diffused-texture-addon/commit/22156213f0c2f599ecdff63b85ea04fcab1c8974))
* **deps:** require urllib3&gt;=2.8.0 to address security advisories ([aa6f4ad](https://github.com/FrederikHasecke/diffused-texture-addon/commit/aa6f4adf01cda0b81d007b948ff7ac6cc1e83165))
* **deps:** require urllib3&gt;=2.8.0 to address security advisories ([936d894](https://github.com/FrederikHasecke/diffused-texture-addon/commit/936d8947c96a3cbe84f251205d8e8f0280e4d2b9))
* **installer:** keep TLS certificate verification for pip installs ([3c2a71e](https://github.com/FrederikHasecke/diffused-texture-addon/commit/3c2a71e09dd36d331f0e1ed9998ca282b2791ca9))
* **installer:** keep TLS certificate verification for pip installs ([321254f](https://github.com/FrederikHasecke/diffused-texture-addon/commit/321254fe38c0a6370beae0be7c17d40056d8eee1))
* **operator:** harden texture generation run state ([23b3b8b](https://github.com/FrederikHasecke/diffused-texture-addon/commit/23b3b8b872895b6b13c944af304cee28d3861e52))
* **operator:** harden texture generation run state ([dc98aaa](https://github.com/FrederikHasecke/diffused-texture-addon/commit/dc98aaa4b1900296dce6b19db3a217f32cf9cc49))
* **pipeline:** enable VAE tiling to avoid SDXL OOM on 12 GB GPUs ([5fbb7fd](https://github.com/FrederikHasecke/diffused-texture-addon/commit/5fbb7fdbda5acba218e543b04b97311ad09dd936))
* **pipeline:** enable VAE tiling to avoid SDXL OOM on 12 GB GPUs ([8e7daaf](https://github.com/FrederikHasecke/diffused-texture-addon/commit/8e7daaf0971a8e3d0489666b496da4df723d9b9f))
* **pipeline:** resolve relative model paths and avoid full VRAM upload ([33b80ec](https://github.com/FrederikHasecke/diffused-texture-addon/commit/33b80ecfe8048e11aa9335b9d3dee723020505ca))
* **pipeline:** resolve relative model paths and avoid full VRAM upload ([807e91a](https://github.com/FrederikHasecke/diffused-texture-addon/commit/807e91a3b069048d9018e6d170142719a807b907))
* **render:** restore the Blender 5 compositor node group after generation ([a924f9f](https://github.com/FrederikHasecke/diffused-texture-addon/commit/a924f9f60a759c791e8edf3112437a137c7275ec))
* **render:** restore the Blender 5 compositor node group after generation ([40c5108](https://github.com/FrederikHasecke/diffused-texture-addon/commit/40c5108c6e5692707e9852b2f47a2e4d43f89da8))
* **texture:** pad UV island edges to prevent visible seams ([a2f74e9](https://github.com/FrederikHasecke/diffused-texture-addon/commit/a2f74e9963b6ebd821b9410e73658b3874e584a4))
* **texture:** pad UV island edges to prevent visible seams ([ba6dc01](https://github.com/FrederikHasecke/diffused-texture-addon/commit/ba6dc01bb1dfe34adac148df06b2ef83175ab53f))
* **uv:** build surface mask from all faces regardless of edit-mode selection ([e09e537](https://github.com/FrederikHasecke/diffused-texture-addon/commit/e09e537bc6c29b26932fcce84f7da589cd37f485))
* **uv:** build surface mask from all faces regardless of edit-mode selection ([098a289](https://github.com/FrederikHasecke/diffused-texture-addon/commit/098a2893b56505a9e0d138ba550719910d17020a))

## [0.3.2](https://github.com/FrederikHasecke/diffused-texture-addon/compare/diffused_texture_addon-v0.3.1...diffused_texture_addon-v0.3.2) (2026-07-19)


### Bug Fixes

* **installer:** pin verified PyTorch backends for Python 3.13 ([3bb46de](https://github.com/FrederikHasecke/diffused-texture-addon/commit/3bb46dec8f98cb8b18d20f707ffbcd96c022ebc8))

## [0.3.1](https://github.com/FrederikHasecke/diffused-texture-addon/compare/diffused_texture_addon-v0.3.0...diffused_texture_addon-v0.3.1) (2026-07-19)


### Bug Fixes

* **render:** scope isolation to active view layer ([7df5ed3](https://github.com/FrederikHasecke/diffused-texture-addon/commit/7df5ed33201b258f748dd94d646366da921d33b5))

## [0.3.0](https://github.com/FrederikHasecke/diffused-texture-addon/compare/diffused_texture_addon-v0.2.0...diffused_texture_addon-v0.3.0) (2026-05-15)


### Features

* **import:** enhance module import handling and add entry point tests ([b810e19](https://github.com/FrederikHasecke/diffused-texture-addon/commit/b810e19e8cda3b45fea1b52dacbd1391094dbe30))
* **texture:** harden runtime validation and render cleanup ([427c417](https://github.com/FrederikHasecke/diffused-texture-addon/commit/427c417606127fe9e0aa2f126ac4767f18c82df6))
* **texture:** harden runtime validation and render cleanup ([a3825d1](https://github.com/FrederikHasecke/diffused-texture-addon/commit/a3825d1c504c1774092ecd1746040e8a3d60a72a))


### Bug Fixes

* **lint:** add installation of Linux system libraries for bpy ([79ee564](https://github.com/FrederikHasecke/diffused-texture-addon/commit/79ee564f0965f05a15a79bfccf2cc38a54c2b835))

## [0.1.0]

- Historical release published before automated semver adoption.

## [0.0.4]

- Historical release published before automated semver adoption.

## [0.0.3]

- Historical release published before automated semver adoption.

## [0.0.2]

- Historical release published before automated semver adoption.

## [0.0.1]

- Historical release published before automated semver adoption.
