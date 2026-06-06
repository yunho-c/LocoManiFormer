# LocoManiFormer

This project is an attempt to reproduce results from the following papers:

- [Deep Whole Body Control by Zipeng Fu et al.](https://manipulation-locomotion.github.io/)

- [LocoFormer by Min Liu et al.](https://generalist-locomotion.github.io/)

- [OpenVLA by Moo Jin Kim et al.](https://openvla.github.io/)

by training a large transformer model for loco-manipulation via RL task with aggressive domain randomization, including different embodiment morphologies (e.g., quadrupedal and bipedal). 

To implement this, we use:

- [Moojoco](https://github.com/Co-Evolve/moojoco), a framework for procedurally generating MuJoCo robots/environments

- The action flow matching head from [$\pi$ model family by Physical Intelligence]()

- An open-source, tractably sized LLM suitable for training, such as DeepSeek / Qwen / SmolLM.
