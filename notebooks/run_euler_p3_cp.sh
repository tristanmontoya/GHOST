#!/bin/bash

# c = c_+
nohup papermill euler_driver.ipynb euler_m04p3cpt1_strong.ipynb -p mach_number 0.4 -p p 3 -p p_geo 3 -p c 'c_+' -p discretization_type 1 -p form 'strong' &>/tmp/papermill_m04p3cpt1_strong.out &

nohup papermill euler_driver.ipynb euler_m04p3cpt1_weak.ipynb -p mach_number 0.4 -p p 3 -p p_geo 3 -p c 'c_+' -p discretization_type 1 -p form 'weak' &>/tmp/papermill_m04p3cpt1_weak.out &

nohup papermill euler_driver.ipynb euler_m04p3cpt2_strong.ipynb -p mach_number 0.4 -p p 3 -p p_geo 3 -p c 'c_+' -p discretization_type 2 -p form 'strong' &>/tmp/papermill_m04p3cpt2_strong.out &

nohup papermill euler_driver.ipynb euler_m04p3cpt2_weak.ipynb -p mach_number 0.4 -p p 3 -p p_geo 3 -p c 'c_+' -p discretization_type 2 -p form 'weak' &>/tmp/papermill_m04p3cpt2_weak.out &

nohup papermill euler_driver.ipynb euler_m04p3cpt3_strong.ipynb -p mach_number 0.4 -p p 3 -p p_geo 3 -p c 'c_+' -p discretization_type 3 -p form 'strong' &>/tmp/papermill_m04p3cpt3_strong.out &

nohup papermill euler_driver.ipynb euler_m04p3cpt3_weak.ipynb -p mach_number 0.4 -p p 3 -p p_geo 3 -p c 'c_+' -p discretization_type 3 -p form 'weak' &>/tmp/papermill_m04p3cpt3_weak.out &

