#!/bin/bash

# c = c_dg
papermill advection_driver.ipynb advection_p3b0c0t1_strong.ipynb -p p 3 -p p_map 1 -p upwind_parameter 0.0 -p c 'c_dg' -p discretization_type 1 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b0c0t1_strong.out &

papermill advection_driver.ipynb advection_p3b0c0t1_weak.ipynb -p p 3 -p p_map 1 -p upwind_parameter 0.0 -p c 'c_dg' -p discretization_type 1 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b0c0t1_weak.out &

papermill advection_driver.ipynb advection_p3b0c0t2_strong.ipynb -p p 3 -p p_map 1 -p upwind_parameter 0.0 -p c 'c_dg' -p discretization_type 2 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b0c0t2_strong.out &

papermill advection_driver.ipynb advection_p3b0c0t2_weak.ipynb -p p 3 -p p_map 1 -p upwind_parameter 0.0 -p c 'c_dg' -p discretization_type 2 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b0c0t2_weak.out &

papermill advection_driver.ipynb advection_p3b0c0t3_strong.ipynb -p p 3 -p p_map 1 -p upwind_parameter 0.0 -p c 'c_dg' -p discretization_type 3 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b0c0t3_strong.out &

papermill advection_driver.ipynb advection_p3b0c0t3_weak.ipynb -p p 3 -p p_map 1 -p upwind_parameter 0.0 -p c 'c_dg' -p discretization_type 3 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b0c0t3_weak.out &

# c = c_+
papermill advection_driver.ipynb advection_p3b0cpt1_strong.ipynb -p p 3 -p p_map 1 -p upwind_parameter 0.0 -p c 'c_+' -p discretization_type 1 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b0cpt1_strong.out &

papermill advection_driver.ipynb advection_p3b0cpt1_weak.ipynb -p p 3 -p p_map 1 -p upwind_parameter 0.0 -p c 'c_+' -p discretization_type 1 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b0cpt1_weak.out &

papermill advection_driver.ipynb advection_p3b0cpt2_strong.ipynb -p p 3 -p p_map 1 -p upwind_parameter 0.0 -p c 'c_+' -p discretization_type 2 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b0cpt2_strong.out &

papermill advection_driver.ipynb advection_p3b0cpt2_weak.ipynb -p p 3 -p p_map 1 -p upwind_parameter 0.0 -p c 'c_+' -p discretization_type 2 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b0cpt2_weak.out &

papermill advection_driver.ipynb advection_p3b0cpt3_strong.ipynb -p p 3 -p p_map 1 -p upwind_parameter 0.0 -p c 'c_+' -p discretization_type 3 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b0cpt3_strong.out &

papermill advection_driver.ipynb advection_p3b0cpt3_weak.ipynb -p p 3 -p p_map 1 -p upwind_parameter 0.0 -p c 'c_+' -p discretization_type 3 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b0cpt3_weak.out &

#!/bin/bash

# c = c_dg
papermill advection_driver.ipynb advection_p3b1c0t1_strong.ipynb -p p 3 -p p_map 1 -p upwind_parameter 1.0 -p c 'c_dg' -p discretization_type 1 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b1c0t1_strong.out &

papermill advection_driver.ipynb advection_p3b1c0t1_weak.ipynb -p p 3 -p p_map 1 -p upwind_parameter 1.0 -p c 'c_dg' -p discretization_type 1 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b1c0t1_weak.out &

papermill advection_driver.ipynb advection_p3b1c0t2_strong.ipynb -p p 3 -p p_map 1 -p upwind_parameter 1.0 -p c 'c_dg' -p discretization_type 2 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b1c0t2_strong.out &

papermill advection_driver.ipynb advection_p3b1c0t2_weak.ipynb -p p 3 -p p_map 1 -p upwind_parameter 1.0 -p c 'c_dg' -p discretization_type 2 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b1c0t2_weak.out &

papermill advection_driver.ipynb advection_p3b1c0t3_strong.ipynb -p p 3 -p p_map 1 -p upwind_parameter 1.0 -p c 'c_dg' -p discretization_type 3 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b1c0t3_strong.out &

papermill advection_driver.ipynb advection_p3b1c0t3_weak.ipynb -p p 3 -p p_map 1 -p upwind_parameter 1.0 -p c 'c_dg' -p discretization_type 3 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b1c0t3_weak.out &

# c = c_+
papermill advection_driver.ipynb advection_p3b1cpt1_strong.ipynb -p p 3 -p p_map 1 -p upwind_parameter 1.0 -p c 'c_+' -p discretization_type 1 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b1cpt1_strong.out &

papermill advection_driver.ipynb advection_p3b1cpt1_weak.ipynb -p p 3 -p p_map 1 -p upwind_parameter 1.0 -p c 'c_+' -p discretization_type 1 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b1cpt1_weak.out &

papermill advection_driver.ipynb advection_p3b1cpt2_strong.ipynb -p p 3 -p p_map 1 -p upwind_parameter 1.0 -p c 'c_+' -p discretization_type 2 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b1cpt2_strong.out &

papermill advection_driver.ipynb advection_p3b1cpt2_weak.ipynb -p p 3 -p p_map 1 -p upwind_parameter 1.0 -p c 'c_+' -p discretization_type 2 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b1cpt2_weak.out &

papermill advection_driver.ipynb advection_p3b1cpt3_strong.ipynb -p p 3 -p p_map 1 -p upwind_parameter 1.0 -p c 'c_+' -p discretization_type 3 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b1cpt3_strong.out &

papermill advection_driver.ipynb advection_p3b1cpt3_weak.ipynb -p p 3 -p p_map 1 -p upwind_parameter 1.0 -p c 'c_+' -p discretization_type 3 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_p3b1cpt3_weak.out &

wait
