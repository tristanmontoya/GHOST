#!/bin/bash

#c = c_dg
nohup papermill euler_driver.ipynb euler_m04p2c0t1_strong.ipynb \ 
-p mach_number 0.4 -p p 2 -p c "c_dg" -p discretization_type 1 -p form "strong" \
&>/tmp/papermill_m04p2c0t1_strong.out &

nohup papermill euler_driver.ipynb euler_m04p2c0t1_weak.ipynb \ 
-p mach_number 0.4 -p p 2 -p c "c_dg" -p discretization_type 1 -p form "weak" \
&>/tmp/papermill_m04p2c0t1_weak.out &

nohup papermill euler_driver.ipynb euler_m04p2c0t2_strong.ipynb \ 
-p mach_number 0.4 -p p 2 -p c "c_dg" -p discretization_type 2 -p form "strong" \
&>/tmp/papermill_m04p2c0t2_strong.out &

nohup papermill euler_driver.ipynb euler_m04p2c0t2_weak.ipynb \ 
-p mach_number 0.4 -p p 2 -p c "c_dg" -p discretization_type 2 -p form "weak" \
&>/tmp/papermill_m04p2c0t2_weak.out &

nohup papermill euler_driver.ipynb euler_m04p2c0t3_strong.ipynb \ 
-p mach_number 0.4 -p p 2 -p c "c_dg" -p discretization_type 3 -p form "strong" \
&>/tmp/papermill_m04p2c0t3_strong.out &

nohup papermill euler_driver.ipynb euler_m04p2c0t3_weak.ipynb \ 
-p mach_number 0.4 -p p 2 -p c "c_dg" -p discretization_type 3 -p form "weak" \
&>/tmp/papermill_m04p2c0t3_weak.out &

#c = c_+
nohup papermill euler_driver.ipynb euler_m04p2cpt1_strong.ipynb \ 
-p mach_number 0.4 -p p 2 -p c "c_+" -p discretization_type 1 -p form "strong" \
&>/tmp/papermill_m04p2cpt1_strong.out &

nohup papermill euler_driver.ipynb euler_m04p2cpt1_weak.ipynb \ 
-p mach_number 0.4 -p p 2 -p c "c_+" -p discretization_type 1 -p form "weak" \
&>/tmp/papermill_m04p2cpt1_weak.out &

nohup papermill euler_driver.ipynb euler_m04p2cpt2_strong.ipynb \ 
-p mach_number 0.4 -p p 2 -p c "c_+" -p discretization_type 2 -p form "strong" \
&>/tmp/papermill_m04p2cpt2_strong.out &

nohup papermill euler_driver.ipynb euler_m04p2cpt2_weak.ipynb \ 
-p mach_number 0.4 -p p 2 -p c "c_+" -p discretization_type 2 -p form "weak" \
&>/tmp/papermill_m04p2cpt2_weak.out &

nohup papermill euler_driver.ipynb euler_m04p2cpt3_strong.ipynb \ 
-p mach_number 0.4 -p p 2 -p c "c_+" -p discretization_type 3 -p form "strong" \
&>/tmp/papermill_m04p2cpt3_strong.out &

nohup papermill euler_driver.ipynb euler_m04p2cpt3_weak.ipynb \ 
-p mach_number 0.4 -p p 2 -p c "c_+" -p discretization_type 3 -p form "weak" \
&>/tmp/papermill_m04p2cpt3_weak.out &
