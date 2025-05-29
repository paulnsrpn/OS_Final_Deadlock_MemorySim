# Simulating and Analyzing Deadlock and Memory Allocation Strategies in Operating Systems
## Description
This project focuses on simulating two key aspects of operating system resource management: **deadlock handling** and **memory allocation**. It implements the **Banker’s Algorithm** to analyze and avoid deadlocks in process scheduling. Additionally, it compares the performance of three memory allocation strategies—**First Fit**, **Best Fit**, and **Worst Fit**—to determine their efficiency in different scenarios. The goal is to visualize and better understand how these strategies work and impact overall system performance through custom simulations.

# Team members and role
Alentajan, Jervie D.     - Visaulization developer
Florida, Emanuel C.      - Tester and Analyst
Platon, Laiza Marie O.   - Documentation Specialist
Serapion, Pauline L.     - Lead Programemer
Sison, Mark Anthony R.   - Leader/Finalization

# Results
The simulation results reveal notable differences among the three memory allocation strategies. In the **First Fit** strategy, processes P0, P1, and P2 were successfully allocated to memory blocks, while P3 was not allocated due to insufficient contiguous space, even though a large enough block existed later in the list. **Best Fit** showed the most efficient use of memory, successfully allocating all four processes (P0 to Block 3, P1 to Block 1, P2 to Block 2, and P3 to Block 4). On the other hand, the **Worst Fit** strategy was less effective, allocating only P0, P1, and P2, with P3 left unallocated, similar to First Fit. These results demonstrate that while First Fit is faster, Best Fit leads to better memory utilization, and Worst Fit may result in inefficient usage depending on block sizes and allocation order.


