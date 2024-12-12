# EN.601.682 Final Project - Group 13: **Semi-Supervised Reinforcement Learning for Autonomous Agents in Dynamic Environments Using CARLA**

## **Introduction**

<p align="center">

<img alt="Demo" src="../.data/ML-DL-Fall2024-ProjectVideo.gif">
<br>Demo Video<br>

</p>

### **Background**

Autonomous navigation in dynamic environments is a significant challenge for robotics and AI, with applications in self-driving vehicles, delivery drones, and robotic exploration. Traditional Reinforcement Learning (RL) approaches require carefully designed reward functions and extensive labeled data, which can lead to undesired behaviors and hinder scalability.

In contrast, semi-supervised and self-supervised learning approaches can efficiently leverage partially labeled or unlabeled data to extract robust representations for downstream RL tasks. This project integrates self-supervised representation learning with curiosity-driven RL to address policy learning in dynamic environments, specifically using the CARLA simulator.

### **Problem Statement**

Learning effective driving policies in urban environments is complex due to:

- Multi-modal sensor inputs (semantic, depth, LiDAR, metadata).
- Limited labeled datasets for training.
- Environmental variability in weather, lighting, and dynamic obstacles.

Our solution leverages a semi-supervised RL framework that incorporates curiosity-driven exploration and self-supervised learning to improve adaptability and decision-making efficiency.

---

## **Methods**

### **Dataset**

The CARLA simulator is used to generate a multi-modal dataset with synchronized sensor data and diverse environmental conditions:

- **Input Modalities**:
  - RGB images (semantic segmentation).
  - Depth images (normalized to 1 channel).
  - LiDAR data (top-down 3-channel projection).
  - Metadata (autopilot commands, weather, GPS).
- **Dynamic Scenarios**:
  - Weather variations (rain, fog, night, sunny).
  - Obstacles, traffic lights, and pedestrians.
  - Multiple urban maps for testing generalization.

The dataset is preprocessed to align multi-modal inputs, making it suitable for representation learning and policy optimization.

---

### **Architecture and Training**

<p align="center">

<img alt="System Overview" src="../.data/Project-Overview.png">
<br>System Overview<br>

</p>

#### Observational Temporal Encoder Architecture -

```mermaid
flowchart TD
    subgraph Inputs
        sem[/"Semantic (1, 420, 680)"/]
        depth[/"Depth (1, 420, 680)"/]
        lidar[/"LiDAR (3, 200, 200)"/]
        meta[/"Metadata (vehicle state, weather)"/]
    end

    subgraph "Semantic Encoder"
        sem --> sem1["Conv2D(1→64) + BN + ReLU"]
        sem1 --> sem2["MaxPool"]
        sem2 --> sem3["ResBlock(64)"]
        sem3 --> sem4["Conv2D(64→128) + BN + ReLU"]
        sem4 --> sem5["ResBlock(128)"]
        sem5 --> sem6["Conv2D(128→256) + BN + ReLU"]
        sem6 --> sem7["ResBlock(256)"]
        sem7 --> sem8["AdaptiveAvgPool + Flatten"]
        sem8 --> sem9["Linear(256→feature_dim)"]
    end

    subgraph "Depth Encoder"
        depth --> d1["Conv2D(1→64) + BN + ReLU"]
        d1 --> d2["MaxPool"]
        d2 --> d3["ResBlock(64)"]
        d3 --> d4["Conv2D(64→128) + BN + ReLU"]
        d4 --> d5["ResBlock(128)"]
        d5 --> d6["Conv2D(128→256) + BN + ReLU"]
        d6 --> d7["ResBlock(256)"]
        d7 --> d8["AdaptiveAvgPool + Flatten"]
        d8 --> d9["Linear(256→feature_dim)"]
    end

    subgraph "LiDAR Encoder"
        lidar --> l1["Conv2D(3→64) + BN + ReLU"]
        l1 --> l2["ResBlock(64)"]
        l2 --> l3["Conv2D(64→128) + BN + ReLU"]
        l3 --> l4["ResBlock(128)"]
        l4 --> l5["Conv2D(128→256) + BN + ReLU"]
        l5 --> l6["ResBlock(256)"]
        l6 --> l7["AdaptiveAvgPool + Flatten"]
        l7 --> l8["Linear(256→feature_dim)"]
    end

    subgraph "Metadata Encoder"
        meta --> m1["Linear(30→256)"]
        m1 --> m2["LayerNorm + ReLU + Dropout"]
        m2 --> m3["Linear(256→512)"]
        m3 --> m4["LayerNorm + ReLU + Dropout"]
        m4 --> m5["Linear(512→feature_dim)"]
    end

    subgraph "Temporal Processing"
        sem9 --> t1["GRU + LayerNorm"]
        d9 --> t2["GRU + LayerNorm"]
        l8 --> t3["GRU + LayerNorm"]
        m5 --> t4["GRU + LayerNorm"]
    end

    subgraph "Feature Fusion"
        t1 & t2 & t3 & t4 --> f1["Concatenate"]
        f1 --> f2["Linear(4×feature_dim→feature_dim×2)"]
        f2 --> f3["LayerNorm + ReLU + Dropout"]
        f3 --> f4["Linear(feature_dim×2→feature_dim)"]
        f4 --> f5["LayerNorm"]
    end

    subgraph "Temporal Fusion"
        f5 --> tf["GRU + LayerNorm"]
    end

    subgraph "Decoders"
        tf --> dec1["Semantic Decoder\nLinear→LayerNorm→ReLU→Linear→Unflatten"]
        tf --> dec2["Depth Decoder\nLinear→LayerNorm→ReLU→Linear→Unflatten"]
        tf --> dec3["LiDAR Decoder\nLinear→LayerNorm→ReLU→Linear→Unflatten"]
    end

    dec1 --> out1["Semantic Reconstruction"]
    dec2 --> out2["Depth Reconstruction"]
    dec3 --> out3["LiDAR Reconstruction"]
```

#### PPO Architecture -

```mermaid
flowchart TD
    %% Inputs
    input[/"Encoder Input\n(semantic, depth, lidar, metadata)"/]---enc[("Observation\nTemporal Encoder")]
    enc---state["Encoded State\n(feature_dim)"]

    %% Policy Network Branch
    state---p1["Linear(feature_dim→hidden_dim)"]
    p1---p2["LayerNorm"]
    p2---p3["Mish"]
    p3---p4["Linear(hidden_dim→hidden_dim)"]
    p4---p5["LayerNorm"]
    p5---p6["Mish"]
    p6---p7["Linear(hidden_dim→action_dim)"]
    p7---p8["Softmax"]
    p8---actions["Action Probabilities"]

    %% Value Network Branch
    state---v1["Linear(feature_dim→hidden_dim)"]
    v1---v2["LayerNorm"]
    v2---v3["Mish"]
    v3---v4["Linear(hidden_dim→hidden_dim)"]
    v4---v5["LayerNorm"]
    v5---v6["Mish"]
    v6---v7["Linear(hidden_dim→1)"]
    v7---value["State Value"]

    %% Adaptive Entropy
    state---ent["Entropy Coefficient\n(learnable parameter)"]
    ent-.->actions

    %% Outputs and Losses
    actions---dist["Categorical Distribution"]
    dist---samp["Sample Actions"]
    dist---lp["Log Probabilities"]
    value---returns["Value Estimates"]

    %% Losses
    samp & lp & returns---loss["PPO Loss Components:\n- Policy Loss\n- Value Loss\n- Entropy Bonus"]

    %% Styling
    classDef input fill:#e1f5fe,stroke:#01579b
    classDef output fill:#e8f5e9,stroke:#1b5e20
    classDef encoder fill:#fff3e0,stroke:#e65100
    classDef loss fill:#fce4ec,stroke:#880e4f
    class input input
    class actions,value output
    class enc encoder
    class loss loss

    %% Layout hints
    subgraph "Policy Network"
        p1
        p2
        p3
        p4
        p5
        p6
        p7
        p8
    end

    subgraph "Value Network"
        v1
        v2
        v3
        v4
        v5
        v6
        v7
    end
```

#### ICM Architecture -

```mermaid
flowchart TD
    %% Inputs
    s1[/"Current State\n(state_dim)"/]---norm1["LayerNorm"]
    s2[/"Next State\n(state_dim)"/]---norm2["LayerNorm"]
    act[/"Action\n(action_dim)"/]---emb["Action\nEmbedding"]

    %% Forward Model Path
    norm1 & emb---concat1["Concatenate"]
    concat1---f1["Linear(state+action→hidden)"]
    f1---f2["LayerNorm"]
    f2---f3["ReLU"]
    f3---f4["Linear(hidden→hidden)"]
    f4---f5["LayerNorm"]
    f5---f6["ReLU"]
    f6---f7["Linear(hidden→state)"]
    f7---pred["Predicted Next State"]

    %% Inverse Model Path
    norm1 & norm2---concat2["Concatenate"]
    concat2---i1["Linear(2×state→hidden)"]
    i1---i2["LayerNorm"]
    i2---i3["ReLU"]
    i3---i4["Linear(hidden→hidden)"]
    i4---i5["LayerNorm"]
    i5---i6["ReLU"]
    i6---i7["Linear(hidden→action)"]
    i7---pred_act["Predicted Action"]

    %% Loss Computation
    pred & norm2---fwd_loss["Forward Loss (MSE)"]
    pred_act & act---inv_loss["Inverse Loss (CrossEntropy)"]
    fwd_loss & inv_loss---total["Total Loss\nβ×forward + (1-β)×inverse"]

    %% Reward Generation
    pred & norm2---rew["Intrinsic Reward\nη×prediction error"]

    %% Styling
    classDef input fill:#e1f5fe,stroke:#01579b
    classDef output fill:#e8f5e9,stroke:#1b5e20
    classDef loss fill:#fce4ec,stroke:#880e4f
    classDef reward fill:#f3e5f5,stroke:#4a148c
    class s1,s2,act input
    class pred,pred_act output
    class fwd_loss,inv_loss,total loss
    class rew reward

    %% Layout hints
    subgraph "Forward Model"
        direction TB
        f1
        f2
        f3
        f4
        f5
        f6
        f7
    end

    subgraph "Inverse Model"
        direction TB
        i1
        i2
        i3
        i4
        i5
        i6
        i7
    end
```

#### 1. **Representation Learning**

We use an **ObservationTemporalEncoder** that processes input sequences across semantic, depth, LiDAR, and metadata channels. The encoder consists of:

- Convolutional and Residual blocks for sensor data.
- Temporal GRU blocks for spatial-temporal features.
- A self-supervised loss combining reconstruction, consistency, compactness, and temporal smoothness.

#### 2. **Pre-Training with PPO and ICM**

We initialize the RL policy using **Proximal Policy Optimization (PPO)** combined with an **Intrinsic Curiosity Module (ICM)**:

- **PPO**: Ensures stable policy updates using clipped objectives.
- **ICM**: Generates intrinsic rewards based on prediction errors in state transitions, encouraging exploration of less predictable states.

The pre-training phase uses Behavioral Cloning (BC) loss to accelerate initial learning.

#### 3. **Active RL Training**

After pre-training, agents are fine-tuned in altered CARLA environments (e.g., new maps, weather conditions). The PPO-ICM framework continues to optimize the policy through real-time interactions with dynamic conditions.

The PPO objective is defined as:
\[
L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)],
\]
where \( \hat{A}_t \) is the Generalized Advantage Estimate.

#### 4. **Evaluation**

We use the **Driving Score Metric** defined as:
\[
\text{Driving Score} = R_i P_i,
\]
where:

- \( R_i \): Route completion percentage.
- \( P_i \): Weighted infraction penalties (e.g., pedestrian collisions, stop sign violations).

| **Infraction Type**           | **Penalty Coefficient** |
|-------------------------------|-------------------------|
| Collision: pedestrians        | 0.50                   |
| Collision: vehicles           | 0.60                   |
| Collision: static objects     | 0.65                   |
| Running: red light            | 0.70                   |
| Running: stop sign            | 0.80                   |

---

## **Results**

- The **ObservationTemporalEncoder** achieved a total test loss of **0.0118**.
- Pre-training was intentionally halted at **40\% BC loss** to focus on exploration efficiency.
- After **1000 episodes** of RL training, the average Driving Score reached **61.6\%** on unseen test maps with altered conditions, calculated using CARLA Leaderboard metrics.

<p align="center">

<img alt="Observation Temporal Encoder Loss Curve" src="../.data/encoder_training_curves.png">
<br>Observation Temporal Encoder Loss Curve<br><br>

<img alt="Pre Training Loss Curve" src="../.data/pre_training_curves.png">
<br>Pre Training Loss Curve<br><br>

<img alt="Pre Training Loss Curve" src="../.data/rl_training_curves_split.png">
<br>Pre Training Loss Curve<br><br>

</p>

---

## **Discussion**

Our findings demonstrate that self-supervised representation learning, combined with curiosity-driven RL, enables agents to adapt to dynamic environments with improved generalization. Key contributions include:

1. Semi-supervised latent feature learning for multi-modal sensor data.
2. Curiosity-driven exploration that reduces reliance on external rewards.
3. Robust performance in unseen environments with dynamic obstacles.

### **Future Directions**

- Explore deeper temporal models for improved sequence processing.
- Incorporate additional intrinsic reward schemes (e.g., count-based exploration).
- Scale experiments to larger CARLA maps with greater variability.

---

## **References**

1. Codevilla, Felipe, et al. "End-to-end driving via conditional imitation learning." ICRA 2018.
2. Jing, Longlong, and Yann LeCun. "Self-supervised learning for video: A survey." TPAMI 2020.
3. Pathak, Deepak, et al. "Curiosity-driven exploration by self-supervised prediction." CVPR 2017 Workshop.
4. Schulman, John, et al. "Proximal policy optimization algorithms." arXiv:1707.06347, 2017.
5. Pomerleau, Dean A. "ALVINN: An autonomous land vehicle in a neural network." NIPS 1989.
6. Kendall, Alex, et al. "Learning to drive in a day." ICRA 2019.
7. Dosovitskiy, Alexey, et al. "CARLA: An open urban driving simulator." CoRL 2017.
8. Liang, Xiaodan, et al. "CIRL: Controllable imitative reinforcement learning for vision-based self-driving." ECCV 2018.
9. Chen, Dian, et al. "Learning to drive from a world on rails." ICCV 2020.
10. Yu, Tianhe, et al. "COMBO: Conservative offline model-based policy optimization." NeurIPS 2021.
11. CARLA Leaderboard Case Study - ALPHA DRIVE, 2024.
12. CARLA Documentation: Driving Benchmark Performance Metrics, 2024.
13. CARLA Autonomous Driving Leaderboard Documentation, 2024.
14. Papers With Code - CARLA Leaderboard Benchmark (Autonomous Driving). Papers With Code, 2024.
15. Yeh, Paul. "Motion Planning on CARLA." GitHub repository, 2020.
16. Pathak, Deepak, Pulkit Agrawal, Alexei A. Efros, and Trevor Darrell. "Curiosity-driven Exploration by Self-supervised Prediction." PMLR, 2017.
17. Zhang, X., et al. "Trajectory-Guided Control Prediction for End-to-End Autonomous Driving." arXiv preprint arXiv:2206.08129v2, 2022.
18. Wu, Penghao, et al. "Trajectory-guided Control Prediction for End-to-end Autonomous Driving." GitHub repository, 2022.
19. Shaikh, Idrees. "Autonomous Driving in CARLA using Deep Reinforcement Learning." GitHub repository, 2021.
20. RobeSafe-UAH. "DQN-CARLA." GitHub repository, 2021.
21. Minelli, John. "Carla-Gym: Multi-Agent RL Interface for CARLA." GitHub repository, 2022.
22. Luca96. "CARLA Driving RL Agent." GitHub repository, 2021.

---
