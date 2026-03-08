# Talos 🐋

**A blazing-fast, zero-allocation local search engine for combinatorial optimization.**

Talos is a high-performance metaheuristic solver built in Rust, specifically tailored for complex resource scheduling and routing tasks—most notably the **Dynamic Berth Allocation Problem (DBAP)**. It is engineered from the ground up using strict **Data-Oriented Design (DOD)** principles to maximize CPU cache locality, defeat the memory allocator, and evaluate tens of millions of candidate solutions per second.
