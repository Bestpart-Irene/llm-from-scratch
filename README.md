# AI Large Model Algorithms — From Principles of Large Models to Training (Fine-tuning) and Practical Deployment

## Course Overview

This comprehensive course covers AI large model algorithms, from fundamental principles to practical training and fine-tuning implementation. The course focuses on DeepSeek and modern large language model technologies.

---

## Course Contents

### Chapter 1: Course Introduction and Environment Setup
- 1-1 Course Introduction and Schedule [Don't Miss]
- 1-2 Why Learn About Large Models and Related Theoretical Knowledge in the AI Era
- 1-3 The Significance of Chinese Developers Learning DeepSeek
- 1-4 Common Learning Resources and Model Downloads for Large Models
- 1-5 Anaconda Introduction and Installation
- 1-6 PyCharm Installation and Remote Server Connection
- 1-7 Following the Map: A Comprehensive Introduction to AI Technology Landscape

### Chapter 2: [Getting Started] DeepSeek Understanding and Experience
- 2-1 The Birth of ChatGPT and DeepSeek's Pursuit
- 2-2 DeepSeek Capability Experience and Impact of Large Models
- 2-3 Getting Started 1 - Building a Powerful Translator Based on DeepSeek Combined with Prompt Engineering
- 2-4 Getting Started 2 - Implementing DeepSeek Distillation Model Private Deployment with Just a Few Lines of Code

### Chapter 3: [Large Model Theory] The Birth Path of DeepSeek
- 3-1 What Problems Does Natural Language Processing Solve
- 3-2 Rule-based and Statistical Methods
- 3-3 Why Use Word Vectors and the Role of Vector Representation
- 3-4 How to Obtain Well-represented Word Vectors?
- 3-5 The Value of Word Vectors
- 3-6 Getting Started: Word Vector Practice
- 3-7 Pre-trained Models (BERT, GPT)
- 3-8 Getting Started: Pre-trained Model Practice
- 3-9 The Birth of Large Language Models
- 3-10 The Birth of DeepSeek
- 3-11 Why Large Models Produce Intelligence

### Chapter 4: [Feature Encoder Transformer] Deep Understanding of Large Model Input and Output
- 4-1 Text Segmentation and Tokens in Large Models
- 4-2 Large Model Tokenizer
- 4-3 Deep Understanding of Tokenizer's Role and Impact
- 4-4 [Getting Started] Tokenizer Practice
- 4-5 Deep Understanding of BPE Algorithm Training and Encoding Process
- 4-6 [Practice] Hand-coding BPE Algorithm Training Code
- 4-7 Initial Understanding of Position Encoding in Large Models
- 4-8 Introduction to Large Model Output Process
- 4-9 Detailed Introduction to Large Model Decoding Principles
- 4-10 [Practice] Finding Optimal Inference Parameters for Large Models Practice

### Chapter 5: [Feature Encoder Transformer] Deep Dive into Attention Mechanism in Transformer
- 5-1 Transformer Basic Knowledge Preparation
- 5-2 [Practice] Hand-coding LayerNorm Code
- 5-3 [Practice] Hand-coding Softmax Code
- 5-4 Deep Understanding of Attention Mechanism
- 5-5 Masked Self-Attention Mechanism
- 5-6 Multi-Head Attention Mechanism
- 5-7 [Practice] Hand-coding Attention Mechanism Code
- 5-8 [Practice] Hand-coding Masked_Self_Attention
- 5-9 [Practice] Hand-coding MaskedMultiHeadAttention Code
- 5-10 Residual Connections and FFN
- 5-11 [Practice] Hand-coding FFN and Residual Structure Implementation Code
- 5-12 [Practice] Hand-coding Transformer Decoder Block Implementation
- 5-13 [Practice] Hand-coding Complete Transformer Code
- 5-14 Evolution of Attention Mechanism: GQA and MQA
- 5-15 [Practice] Hand-coding MQA Attention Mechanism Code
- 5-16 [Practice] Hand-coding GQA Attention Mechanism Code

### Chapter 6: [Feature Encoder Transformer] Deep Dive into Position Encoding in Transformer
- 6-1 Introduction to Relative Position Encoding
- 6-2 Rotary Position Encoding Theory
- 6-3 Hand-coding Rotary Position Encoding ROPE
- 6-4 Core Parameters of Rotary Position Encoding and Their Impact
- 6-5 Variants of Rotary Position Encoding

### Chapter 7: [Pre-training] Pre-training of Large Language Models
- 7-1 Introduction to Classic Training Framework for Large Language Models
- 7-2 Large Model Pre-training (Objectives and Tasks)
- 7-3 Large Model Pre-training (MTPL Multi-Token Prediction)
- 7-4 Large Model Pre-training (Pre-training Data and Processing)
- 7-5 Large Model Pre-training (Pre-training Process)
- 7-6 Large Model Evaluation System
- 7-7 Evaluation of Large Model Code Capabilities
- 7-8 Evaluation of Large Model Math Capabilities
- 7-9 Evaluation of Large Model Reading Comprehension Capabilities
- 7-10 Introduction to Large Model Comprehensive Evaluation Leaderboards
- 7-11 [Practice] DeepSeek Code Capability Evaluation Practice - Step 1: Model Inference
- 7-12 [Practice] DeepSeek Code Capability Evaluation Practice - Step 2: Model Result Processing
- 7-13 [Practice] DeepSeek Code Capability Evaluation - Step 3: Code Execution Check

### Chapter 8: [Pre-training] Data Engineering for Pre-training
- 8-1 Large Model Pre-training Data Collection Process - Dataset 1
- 8-2 Large Model Pre-training Data Collection Process - Dataset 2
- 8-3 Large Model Pre-training Data Collection Process - Pre-training Data Construction Workflow
- 8-4 Large Model Pre-training Data Processing - Data Processing Workflow
- 8-5 Large Model Pre-training Data Processing - Data Filtering + Deduplication + Review
- 8-6 Multi-domain Data Proportioning and Learning Sequence for Large Language Model Pre-training
- 8-7 Large Model Security Issues
- 8-8 Large Model Security Challenges: New Attacks and Defense
- 8-9 In-depth Analysis of DoReMI for LLM Pre-training Domain Data Proportioning

### Chapter 9: [Pre-training] Hardware System Explanation for Pre-training
- 9-1 Introduction to Distributed Training Clusters for Large Models
- 9-2 Distributed Training Clusters for Large Models - Parameter Server Architecture and Decentralized Architecture
- 9-3 Computing Power Foundation for Large Models - In-depth Analysis of GPU Structure
- 9-4 Computing Power Foundation for Large Models - NVIDIA Hardware Architecture System (Fermi Architecture)
- 9-5 Computing Power Foundation for Large Models - NVIDIA Hardware Architecture System (Ampere Architecture)
- 9-6 Large Model Training Communication Efficiency Enhancement with NVLink
- 9-7 Large Model Training Communication Efficiency Enhancement with Topology Structure
- 9-8 Large Model Distributed Training Communication Protocols NCCL, GRPC, HTTP

### Chapter 10: [Pre-training] Large Model Distributed Pre-training Process
- 10-1 Large Model Distributed Training Overview - Pre-training Tasks and Loss Functions
- 10-2 [Practice] Hand-coding Cross-Entropy Loss Function Code
- 10-3 Large Model Distributed Training - Data Parallelism
- 10-4 Large Model Distributed Training - Model Parallelism Overview
- 10-5 Large Model Distributed Training Model Parallelism - Embedding Layer Parallelism
- 10-6 Embedding Parallelism Code Analysis
- 10-7 Model Parallelism - Deep Understanding of Matrix Multiplication Parallelism Principles
- 10-8 Model Parallelism - Deep Understanding of Matrix Multiplication Parallelism Code Analysis
- 10-9 Model Parallelism - Deep Understanding of Cross-Entropy Loss Parallelism Principles
- 10-10 Model Parallelism - Deep Understanding of Cross-Entropy Loss Parallelism Code
- 10-11 Model Parallelism - Deep Understanding of Pipeline Parallelism
- 10-12 Distributed Training - Heterogeneous System Parallelism
- 10-13 Large Model Training Memory Usage Analysis
- 10-14 Distributed Training Software Framework DeepSpeed
- 10-15 DeepSpeed Zero DP Stage
- 10-16 DeepSpeed Zero Offload

### Chapter 11: [Post-training] Supervised Fine-tuning
- 11-1 Pre-training and Post-training of Large Language Models
- 11-2 Instruction Fine-tuning Technology for Large Language Models
- 11-3 Evaluation Methods for Large Model Fine-tuning
- 11-4 Data Construction for Large Language Model Fine-tuning
- 11-5 Chain-of-Thought Data in Large Language Models
- 11-6 Large Language Model Fine-tuning Framework LlamaFactory
- 11-7 llama_factory Full Parameter Fine-tuning Practice

### Chapter 12: [Post-training] Parameter Efficient Fine-tuning
- 12-1 Overview of Parameter Efficient Fine-tuning
- 12-2 Deep Understanding of LoRA Parameter Efficient Fine-tuning Principles
- 12-3 Deep Understanding of Prefix Tuning and Prompt Tuning
- 12-4 Deep Understanding of Adapter Tuning
- 12-5 LoRA Parameter Efficient Fine-tuning Practice

### Chapter 13: [Post-training] Reinforcement Learning Basics
- 13-1 Overview of Reinforcement Learning from Human Feedback
- 13-2 Basic Concepts of Reinforcement Learning + Policy
- 13-3 Basic Concepts of Reinforcement Learning + Value Function
- 13-4 Introduction to Bellman Equation
- 13-5 Stochastic Policy Gradient Algorithm
- 13-6 [Practice] Reinforcement Learning Practice - Environment Modeling
- 13-7 [Practice] Reinforcement Learning Practice - Policy Evaluation
- 13-8 [Practice] Reinforcement Learning Practice - Policy Optimization

### Chapter 14: [Post-training] Reinforcement Learning from Human Feedback
- 14-1 Introduction to Reward Model
- 14-2 Detailed Explanation of PPO Algorithm
- 14-3 Detailed Explanation of PPO Algorithm Training Process
- 14-4 Hand-coding Reward Model Code
- 14-5 Deep Understanding of PPO Algorithm Code

### Chapter 15: [DeepSeek Core Technology Revealed] The Rise of Domestic AI - DeepSeek Core Technology Breakthroughs
- 15-1 Introduction to DeepSeek Model Innovations
- 15-2 KV Cache
- 15-3 Deep Understanding of MLA Mechanism and Principles
- 15-4 Hand-coding MLA Implementation Code
- 15-5 Deep Understanding of MoE Mechanism and Principles
- 15-6 Hand-coding MoE Implementation Code

### Chapter 16: [Large Model Logical Reasoning Capability] DeepSeek V3 and DeepSeek R1
- 16-1 Deep Understanding of Large Model Logical Reasoning Capability
- 16-2 Deep Understanding of Chain-of-Thought Technology
- 16-3 Deep Understanding of GRPO Algorithm
- 16-4 GRPO Practice Project Task Introduction
- 16-5 Auxiliary Function Implementation
- 16-6 Reward Function Design and Implementation
- 16-7 Data Loading and Processing
- 16-8 Hand-coding GRPO Training Code

### Chapter 17: [Enterprise Implementation Practice] Empowering All Industries: Large Model Implementation Application Analysis
- 17-1 Detailed Explanation of Large Model Implementation Application Capabilities
- 17-2 Introduction to Core Methodology for Large Model Implementation Applications
- 17-3 Introduction to Typical Scenarios and Cases for Large Model Implementation Applications
- 17-4 Challenges of Large Model Implementation Applications

### Chapter 18: [Enterprise Implementation Practice] Core Methodology for Large Model Implementation Applications
- 18-1 Deep Understanding of Prompt Engineering and Methodology
- 18-2 Deep Understanding of RAG System
- 18-3 Understanding the Model Quantization Process
- 18-4 Large Model Inference Acceleration and vLLM
- 18-5 Basic Principles of Large Model Agent
- 18-6 Model Regular Inference Practice
- 18-7 vLLM Inference Acceleration Practice

### Chapter 19: [Enterprise Implementation Practice] Agent Intelligent Government Assistant: Capable of Understanding 100,000-Character Long Documents
- 19-1 Project Goals and Requirements
- 19-2 Source Data Collection
- 19-3 Text Vectorization
- 19-4 Text Vectorization Practice
- 19-5 Database ES Introduction and Installation
- 19-6 ES Mapping Construction and Index Creation
- 19-7 ES Data Writing
- 19-8 ES Data Deletion
- 19-9 Writing Project Data, Embedding and Text Data
- 19-10 In-depth Introduction to ES Retrieval DSL
- 19-11 Hand-writing Naive RAG
- 19-12 Introduction to Gradio Chat Dialog Box
- 19-13 Gradio Hello
- 19-14 Building Gradio Large Model Dialog Window
- 19-15 Building RAG Streaming Output Pipeline
- 19-16 Efficient Training Data Synthesis
- 19-17 Synthesizing Government-related Training Data
- 19-18 Generating Government Process Training Data and Optimization
- 19-19 Centralized Training Data Processing
- 19-20 Model Fine-tuning - Constructing Training and Test Sets
- 19-21 Model Fine-tuning - Starting Training
- 19-22 Model Training Effect Testing
- 19-23 Model Training Results Evaluation

### Chapter 20: [Enterprise Implementation Practice] Official Document Writing System: Capable of Hierarchical Multi-level Directory 10,000-Character Long Document Writing
- 20-1 Official Document Generation Project Requirements Analysis
- 20-2 Project Requirements Breakdown and Prompt Construction
- 20-3 Official Document Data Collection
- 20-4 Training Data Construction
- 20-5 Starting Official Document Large Model Training
- 20-6 Building Model Evaluation Set and Model Evaluation Practice
- 20-7 Model Training Effect Analysis Practice
- 20-8 Model Problem Solving, Optimization Analysis and Targeted Data Construction
- 20-9 New Training Data Synthesis and Model Training
- 20-10 Final Model Effect Evaluation
- 20-11 Official Document Writing Large Model Summary

### Chapter 21: [Outlook and Prospects] Multimodal and Large Model Development Trends
- 21-1 Development Trends of Large Models
- 21-2 Theoretical Introduction to Multimodal Large Models
- 21-3 Course Summary and Career Guidance

---

# 🌟 LLM / DeepSeek Full-Stack Skill Tree

> From **fundamental principles → large-scale model training → engineering deployment → enterprise-level production projects**  
> A complete growth roadmap for **LLMs, DeepSeek, reasoning models, AI agents, and enterprise intelligent systems**

---

## 📚 Phase 1: Fundamental Theory & Core Technologies

### 🧠 Theoretical Knowledge

- Neural Network Fundamentals  
- Deep Learning Fundamentals  
- Reinforcement Learning Fundamentals  
- Transformer Principles (Attention Mechanism, etc.)
- MoE, MLA and other DeepSeek core features
- DeepSeek-R1 Logical Reasoning Model Principles

---

### ⚙️ Core Mechanisms

- LLM Tokenizer  
- Positional Encoding & RoPE (Rotary Position Embedding)  
- LLM Output Decoding Process  
- PagedAttention Mechanism  
- GPTQ Model Quantization Techniques  
- Distributed Training & Low-level Operators  

---

### 💻 Hands-on Coding (From Scratch)

- Implement Attention from Scratch  
- Implement Transformer from Scratch  
- Implement RoPE from Scratch  
- Implement BPE Algorithm from Scratch  
- Implement MoE from Scratch  
- Implement MLA from Scratch  
- Implement Model Parallelism (TP, PP) from Scratch  
- Word Embedding Theory and Practice  
- LLM Positional Encoding Practice  

---

## 🔧 Phase 2: Training & Fine-tuning Techniques

### 🏗️ Training Methods

- LLM Pre-training Principles  
- Distributed Training Frameworks (DeepSpeed / Megatron)
- End-to-end LLM Pre-training with Massive Data Processing  
- Hundred-billion Token Scale Data Processing Pipeline  
- Detailed Walkthrough of Hundred-billion Parameter & Trillion Token Pre-training Code  
- LLM Training Data Construction Methods  

---

### 🎯 Fine-tuning Techniques

- Supervised Fine-tuning (SFT) Techniques  
- Efficient Parameter Fine-tuning (LoRA) Principles  
- Reinforcement Fine-tuning Techniques & Framework (TRL)  
- GRPO Algorithm Reinforcement Fine-tuning Practice  

---

### 🧩 Training Framework Practice

- Supervised Fine-tuning with LLaMA-Factory  
- Efficient Parameter Fine-tuning with PEFT  
- Fine-tuning with Distilled DeepSeek-R1 CoT Data  
  - Data Construction  
  - Model Training  
  - Model Evaluation  

---

### ⚡ Optimization Techniques

- INT8 / INT4 Quantization Principles  
- Quantization-aware Distillation  
- Model Deployment under Limited Resources (INT8 / INT4)  

---

## 🚀 Phase 3: Development & Deployment

### 🧠 Core Skills

- Retrieval-Augmented Generation (RAG)  
- AI Agents & Agentic Development Paradigm  
- Vector Databases  
- Document Retrieval & Ranking  
- LLM Quantization & Deployment  
- High-performance Inference Deployment (vLLM)  

---

### 🛠️ Practical Projects

- DeepSeek Toolchain  
- Translation Tools with DeepSeek + Prompt Engineering  
- LLM Text Generation Practice  
- LLM Document Understanding Practice  
- LLM Capability Distillation Practice  
- Private Deployment of Distilled DeepSeek Models  
- Document Vectorization Pipelines  

---

## 💼 Phase 4: Enterprise-Level Landing Projects

---

## 🏛️ Project 1: Government Affairs Intelligent Assistant Agent System

### 📌 Project Overview

An enterprise-grade intelligent assistant system optimized for **government affairs scenarios**, combining **LLM + Agent architecture**.

Key capabilities:

- Parsing documents up to **100,000 characters**
- Accurate extraction of key information from complex policies and regulations
- Automatic structured summaries via semantic analysis
- Knowledge base construction and retrieval
- Long-document segmentation and reasoning

---

### 🧠 Skills Acquired

- Project Framework Setup  
- Prompt Design & Optimization  
- Document Retrieval & Ranking  
- Agent Assistant Construction  
- Knowledge Base Construction  
- Business Problem Analysis & Decomposition  
- Model Evaluation & Bad Case Analysis  
- Model Quantization & Distributed Inference  
- LLM Function Call Capabilities  
- Training Data Construction  
- Long Document Analysis & Understanding  

---

## 📝 Project 2: Intelligent Official Document Writing System

### 📌 Project Overview

An intelligent writing system designed for **high-frequency official document scenarios**, deeply integrating **official document standards** with generative AI.

Capabilities:

- Structured automatic document creation
- Supports long documents up to **10,000 characters**
- Reference-based and outline-driven generation
- Enterprise-ready deployment

---

### 🧠 Skills Acquired

- Project Framework Setup & Prompt Design  
- Long-text Writing Solution Design  
- Reference-based Writing Solution Design  
- Long-text Data Construction  
- Text Generation Model Training  
- Multi-level Directory Training Data Construction  
- Generative Effect Evaluation  
- Model Quantization & Distributed Inference  
- One-click Deployment Scripts  
- Resource Scheduling & Operations  

---

### 🔄 Functional Workflow

1. Input Title  
2. Content Summary & References  
3. Outline Generation & Manual Adjustment  
4. Long Document Generation  

---

## ✅ Learning Outcome

After completing this skill tree, you will be able to:

- Understand **LLM & DeepSeek internals** at system level  
- Train and fine-tune **large-scale models**  
- Deploy models efficiently under **resource constraints**  
- Build **Agent-based enterprise AI systems**  
- Independently deliver **government & enterprise-grade AI solutions**

---

🚀 *This roadmap is designed for engineers aiming at **LLM Research, Infra, Applied AI, and Enterprise AI Architect roles***  


