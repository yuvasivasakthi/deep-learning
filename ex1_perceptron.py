import numpy as np
x1=np.array([0,0,1,1])
x2=np.array([0,1,0,1])
y=np.array([0,1,1,1])
epochs=int(input("enter the epoch:"))
bias=1
l=0.4
a=[]
a.append(float(input("enter the weights for bias:")))
a.append(float(input("enter the weights for x1:")))
a.append(float(input("enter the weights for x2:")))
w=np.array(a)
n=x1.shape[0]
for k in range(epochs):
    for i in range(n):
        f=x1[i]*w[1]+x2[i]*w[2]+bias*w[0]
        y_out=(f>0).astype(int)
        error=y[i]-y_out
        if error!=0:
            w[1]=w[1]+l*error*x1[i]
            w[2]=w[2]+l*error*x2[i]
            w[0]=w[0]+l*error*bias
        print("-------")
        print("updated weights after",i+1,"input instance")
        print("x1 weight",w[1])
        print("x2 weight",w[2])
        print("bias weight",w[0])
print("-------")
print("final weight ",epochs,"epochs(s)")
print("updated weight for x1",w[x1])
print("updated weight for x2",w[x2])
print("updated weight for bias",w[0])
