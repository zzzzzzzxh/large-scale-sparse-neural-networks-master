#fx=x^3-2x+4

h=0.25
x__2=1
x__1=1.25
x_0=1.5
x_1=1.75
x_2=2

f_x_0 = pow(x_0,3) -2*x_0+4

f_x_1 = pow(x_1,3) -2*x_1+4

f_x_2 = pow(x_2,3) -2*x_2+4

forward=(f_x_2-f_x_1)/(x_2-x_1)

first_derivation=3*pow(x_1,2)-2

error_forward = (first_derivation-forward)/first_derivation

backward=(f_x_1-f_x_0)/(x_1-x_0)

error_backward=(first_derivation-backward)/first_derivation

central =(f_x_2-f_x_0)/(x_2-x_0)

error_central=(first_derivation-central)/first_derivation

#second dervation

f_x__1 = pow(x__1,3) -2*x__1+4

f_x__2 = pow(x__2,3) -2*x__2+4

second_forward=(f_x_2-2*f_x_1+f_x_0)/pow(h,2)

second_derivation=6*x_0

error_forward_second=abs((second_derivation-second_forward)/second_derivation)

second_backward=(f_x_0-2*f_x__1+f_x__2)/pow(h,2)

error_backward_second=abs((second_derivation-second_backward)/second_derivation)

second_central=(f_x_1-2*f_x_0+f_x__1)/pow(h,2)

error_central_second=abs((second_derivation-second_central)/second_derivation)

