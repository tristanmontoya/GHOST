#!/bin/bash

# c = c_dg
papermill euler_driver.ipynb euler_m04p4c0t1_strong.ipynb -p mach_number 0.4 -p p 4 -p p_map 4 -p c 'c_dg' -p discretization_type 1 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_m04p4c0t1_strong.out &

papermill euler_driver.ipynb euler_m04p4c0t1_weak.ipynb -p mach_number 0.4 -p p 4 -p p_map 4 -p c 'c_dg' -p discretization_type 1 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_m04p4c0t1_weak.out &

papermill euler_driver.ipynb euler_m04p4c0t2_strong.ipynb -p mach_number 0.4 -p p 4 -p p_map 4 -p c 'c_dg' -p discretization_type 2 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_m04p4c0t2_strong.out &

papermill euler_driver.ipynb euler_m04p4c0t2_weak.ipynb -p mach_number 0.4 -p p 4 -p p_map 4 -p c 'c_dg' -p discretization_type 2 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_m04p4c0t2_weak.out &

papermill euler_driver.ipynb euler_m04p4c0t3_strong.ipynb -p mach_number 0.4 -p p 4 -p p_map 4 -p c 'c_dg' -p discretization_type 3 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_m04p4c0t3_strong.out &

papermill euler_driver.ipynb euler_m04p4c0t3_weak.ipynb -p mach_number 0.4 -p p 4 -p p_map 4 -p c 'c_dg' -p discretization_type 3 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_m04p4c0t3_weak.out &

# c = c_+
papermill euler_driver.ipynb euler_m04p4cpt1_strong.ipynb -p mach_number 0.4 -p p 4 -p p_map 4 -p c 'c_+' -p discretization_type 1 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_m04p4cpt1_strong.out &

papermill euler_driver.ipynb euler_m04p4cpt1_weak.ipynb -p mach_number 0.4 -p p 4 -p p_map 4 -p c 'c_+' -p discretization_type 1 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_m04p4cpt1_weak.out &

papermill euler_driver.ipynb euler_m04p4cpt2_strong.ipynb -p mach_number 0.4 -p p 4 -p p_map 4 -p c 'c_+' -p discretization_type 2 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_m04p4cpt2_strong.out &

papermill euler_driver.ipynb euler_m04p4cpt2_weak.ipynb -p mach_number 0.4 -p p 4 -p p_map 4 -p c 'c_+' -p discretization_type 2 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_m04p4cpt2_weak.out &

papermill euler_driver.ipynb euler_m04p4cpt3_strong.ipynb -p mach_number 0.4 -p p 4 -p p_map 4 -p c 'c_+' -p discretization_type 3 -p form 'strong' &>/scratch/z/zingg/tmontoya/tmp/papermill_m04p4cpt3_strong.out &

papermill euler_driver.ipynb euler_m04p4cpt3_weak.ipynb -p mach_number 0.4 -p p 4 -p p_map 4 -p c 'c_+' -p discretization_type 3 -p form 'weak' &>/scratch/z/zingg/tmontoya/tmp/papermill_m04p4cpt3_weak.out &

wait

