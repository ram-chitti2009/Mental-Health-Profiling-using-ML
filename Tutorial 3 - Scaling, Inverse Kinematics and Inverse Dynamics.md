The aim of the tutorial is to show opensim's ability to solve an inverse kinematics and inverse dynamics problem using experimental data.



This is sorta like reverse engineering-

so you are given the experimental data of actual gait- and inverse kin allows you to compute the joint angles, and inverse dynamics allows you to compute net joint reactions, and net joint movements that produce that experimental gait.



**Key Terms :**

Inverse kinematics computes the joint angles for a musculoskeletal model that best reproduce the motion of a subject.



Inverse Dynamics then use joint angles; angular velocities and angular accelerations of the mode, together with experimental groud reaction forces and movements to solve for the net reaction forces and moments at each of the joints.



Subject specific modeling involves :

1. Scaling a generic musculoskeletal model to modify its anthropometry or physical dimensions so that it matches the anthropometry of the specific subject whose gait we tryna model.
2. Registering the markers on the generic model to match the markers placed on the subject during data collection.

Scaling is performed using a combo of two methods :

Manual Scaling : this allows the user to scale a segment based on some predetermined scale factor. manual scaling is needed when you're doing something niche, and suitable data is not available.

Measurement based scaling : scaling that determines the scale factors for a specific body segment by computing the distance between specified markers on the model, and corresponding experimental marker positions(what was used during the experiment on the subject). So essentially, you compare the distance between the generic model markers, and subject specific markers, and use that to compute your scale factor.



Kinematics is the study of motion without considering the factors that produce that motion(we do not consider the forces or moments that produce the specified motion). The purpose of IK(Inverse Kinematics) is to examine and figure out the joint angles that cause that particular motion with the help of experimental data.



Procedure :

For each time step of the lab-recorded motion data, IK figures out a set of joint angles that produce the pose/motion at that specific time point's experimental kinematics.

OpenSim does this through a mathematical equation(weighted least squares optimization problem with a loss function of marker error).



Marker Error is basically the distance between the experimental marker and the corresponding model-generic marker. Each marker holds a weight aka how strongly the marker's error term shud be minimized in the least squares problem to achieve the experimental gait.



in short - Find

q that minimizes the total weighted distance between model markers and experimental markers.



**Inverse Dynamics : Study of motion and forces and the moments that produce motion. The purpose is to estimate the forces and moments what produced the particular motion. this results will help us infer how the muscles were used in this motion. Equations are solved iteratively.**





