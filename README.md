# Branch-Price-Cut-and-Switch (BPC&S) for Team Formation and Routing for Airport Baggage Handling Tasks

This branch is based on the master branch and contains a rolling horizon implementation, designed to solve large-scale
(such as full-day) instances. The approach roughly works as follows:

- segment the planning horizon into intervals of identical length
- sequentially solve the planning problem for each interval separately, taking the decisions (and thus, worker availabilities) of the prior intervals into account

