---
slideOptions:
  transition: slide
---
<style>
.reveal {
  font-size: 24px;
}
</style>


# Trajectory planning

Summary from the book of [Professor Peter Corke](https://link.springer.com/book/10.1007%2F978-3-642-20144-8) and [John J. Craig](http://www.mech.sharif.ir/c/document_library/get_file?uuid=5a4bb247-1430-4e46-942c-d692dead831f&groupId=14040)

For better look check this note in [HackMD.io](https://hackmd.io/@libernormous/trajectory-planning)

by:
[Liber Normous](https://hackmd.io/@libernormous)

---

# Intro
+ it is explained on Professor Peter Corke [[1]](https://link.springer.com/book/10.1007%2F978-3-642-20144-8) book page 44 and John J. Craig [[2]](http://www.mech.sharif.ir/c/document_library/get_file?uuid=5a4bb247-1430-4e46-942c-d692dead831f&groupId=14040) page 207
+ Path is a way from one point to another point
+ Trajectory is a path with a specific timing

---

# Smooth One-Dimensional Trajectories
Smoothness in this context means that its first few temporal derivatives are **continuous**. Typically velocity and acceleration are required to be continuous and sometimes also the derivative of acceleration or jerk

----

## Ultimate matrix

$$
\left(\begin{matrix}
s_0\\
s_T\\
\dot{s_0}\\
\dot{s_T}\\
\ddot{s_0}\\
\ddot{s_T}\\
\end{matrix}\right)=
\left(\begin{matrix}
0 & 0 & 0 & 0 & 0 & 1\\
T^5 & T^4 & T^3 & T^2 & T & 1\\
0 & 0 & 0 & 0 & 1 & 0\\
5T^4 & 4T^3 & 3T^2 & 2T & 1 & 0\\
0 & 0 & 0 & 2 & 0 & 0\\
20T^3 & 12T^2 & 6T & 2 & 0 & 0\\
\end{matrix}\right)
\left(\begin{matrix}
A\\
B\\
C\\
D\\
E\\
F\\
\end{matrix}\right)
$$

----

## Design steps

1. Decide your initial position $s_0$ and final position $s_T$
2. Decide your initial velocity $\dot{s_0}$ and final velocity $\dot{s_T}$
3. Decide your initial acceleration $\ddot{s_0}$ and final acceleration $\ddot{s_T}$
4. Find coefficient $A\ B\ C\ D\ E\ F$
5. How? inverse the matrix of course!
6. Note: this step is only for 1 axis, repeat the process for another axis
7. Note: $s_0$ is only for when $t=0$, and $s_t$ is only for when $t=T$

----

## Actuation step

$$
\left(\begin{matrix}
s(t)\\
\dot{s(t)}\\
\ddot{s(t)}\\
\end{matrix}\right)=
\left(\begin{matrix}
t^5 & t^4 & t^3 & t^2 & t & 1\\
5t^4 & 4t^3 & 3t^2 & 2t & 1 & 0\\
20t^3 & 12t^2 & 6t & 2 & 0 & 0\\
\end{matrix}\right)
\left(\begin{matrix}
A\\
B\\
C\\
D\\
E\\
F\\
\end{matrix}\right)
$$

Depends on what you want. Do you want to actuate using
1. position $s(t)$ (for direct animation)
2. speed $\dot{s(t)}$ (for motor)
3. acceleration $\ddot{s(t)}$ (IDK)

----

## CODE

```python=0
import NumPy as np
import matplotlib.pyplot as plt

# %%
def traj_poly(s0,stf,sd0,sdtf,sdd0,sddtf,t):
    t0=t[0] # NOTe! t0 must always 0
    tf=t[-1]
    if t0 != 0:
        return 0
    #solving for equation
    coef = np.zeros((6,1)) #we are looking for this
    param = np.asarray([[s0],[stf],[sd0],[sdtf],[sdd0],[sddtf]])
    mat = np.asarray([[0,0,0,0,0,1],
            [tf**5,tf**4,tf**3,tf**2,tf,1],
            [0,0,0,0,1,0],
            [5*tf**4,4*tf**3,3*tf**2,2*tf,1,0],
            [0,0,0,2,0,0],
            [20*tf**3,12*tf**2,6*tf,2,0,0]])
    mat_i = np.linalg.inv(mat) #inverse
    coef = np.matmul(mat_i,param) #acquiring A B C D E F

    #using equation
    zeros = np.zeros(t.shape)
    ones = np.ones(t.shape)
    twos = ones*2
    mat = np.asarray([ #the original equation
        [t**5,t**4,t**3,t**2,t,ones],
        [5*t**4,4*t**3,3*t**2,2*t,ones,zeros],
        [20*t**3,12*t**2,6*t,twos,zeros,zeros]
    ])
    coef_tensor=(np.repeat(coef,t.size,axis=1))
    coef_tensor=np.reshape(coef_tensor,(coef_tensor.shape[0],1,coef_tensor.shape[1]))
    # d = np.tensordot(mat,coef_tensor,axes=[1, 0]).diagonal(axis1=1, axis2=3) #alternative way
    res = np.einsum('mnr,ndr->mdr', mat, coef_tensor)
    return res
```

----

## Example

CODE:
```python=0
# %%
t = np.arange(1,11)
y = traj_poly(0,1,0,0,0,0,t)
plt.plot(y[0,0,:],'r',y[1,0,:],'g',y[2,0,:],'b')
```

![](https://i.imgur.com/2opNhWJ.png)

----

## Shift the time

To shift a signal $f(x)=a+bx$ to the right (delay), we can do $f(x-\tau)=a+b(x-\tau)$ with $\tau$ as the amount of delay. Our matrix became more general like this:

$$
\left(\begin{matrix}
s_0\\
s_T\\
\dot{s_0}\\
\dot{s_T}\\
\ddot{s_0}\\
\ddot{s_T}\\
\end{matrix}\right)=
\left(\begin{matrix}
T_0^5 & T_0^4 & T_0^3 & T_0^2 & T_0 & 1\\
T_f^5 & T_f^4 & T_f^3 & T_f^2 & T_f & 1\\
5T_0^4 & 4T_0^3 & 3T_0^2 & 2T_0 & 1 & 0\\
5T_f^4 & 4T_f^3 & 3T_f^2 & 2T_f & 1 & 0\\
20T_0^3 & 12T_0^2 & 6T_0 & 2 & 0 & 0\\
20T_f^3 & 12T_f^2 & 6T_f & 2 & 0 & 0\\
\end{matrix}\right)
\left(\begin{matrix}
A\\
B\\
C\\
D\\
E\\
F\\
\end{matrix}\right)
$$

----

Then you can actuate with $T_0<t<T_f$

$$
\left(\begin{matrix}
s(t)\\
\dot{s(t)}\\
\ddot{s(t)}\\
\end{matrix}\right)=
\left(\begin{matrix}
t^5 & t^4 & t^3 & t^2 & t & 1\\
5t^4 & 4t^3 & 3t^2 & 2t & 1 & 0\\
20t^3 & 12t^2 & 6t & 2 & 0 & 0\\
\end{matrix}\right)
\left(\begin{matrix}
A\\
B\\
C\\
D\\
E\\
F\\
\end{matrix}\right)
$$

----

CODE:
```python=0
def traj_poly2(s0,stf,sd0,sdtf,sdd0,sddtf,t0,tf,step=1):
    #arranging time
    t = np.arange(t0,tf+step,step)

    #solving for equation
    coef = np.zeros((6,1)) #we are looking for this
    param = np.asarray([[s0],[stf],[sd0],[sdtf],[sdd0],[sddtf]])
    mat = np.asarray([
        [t0**5,t0**4,t0**3,t0**2,t0,1],
        [tf**5,tf**4,tf**3,tf**2,tf,1],
        [5*t0**4,4*t0**3,3*t0**2,2*t0,1,0],
        [5*tf**4,4*tf**3,3*tf**2,2*tf,1,0],
        [20*t0**3,12*t0**2,6*t0,2,0,0],
        [20*tf**3,12*tf**2,6*tf,2,0,0]
        ])
    mat_i = np.linalg.inv(mat) #inverse
    coef = np.matmul(mat_i,param) #acquiring A B C D E F

    #using equation
    zeros = np.zeros(t.shape)
    ones = np.ones(t.shape)
    twos = ones*2
    mat = np.asarray([ #the original equation
        [(t)**5,(t)**4,(t)**3,(t)**2,(t),ones],
        [5*(t)**4,4*(t)**3,3*(t)**2,2*(t),ones,zeros],
        [20*(t)**3,12*(t)**2,6*(t),twos,zeros,zeros]
    ])
    coef_tensor=(np.repeat(coef,t.size,axis=1))
    coef_tensor=np.reshape(coef_tensor,(coef_tensor.shape[0],1,coef_tensor.shape[1]))
    # d = np.tensordot(mat,coef_tensor,axes=[1, 0]).diagonal(axis1=1, axis2=3) #alternative way
    res = np.einsum('mnr,ndr->mdr', mat, coef_tensor)

    time  = t
    possi = res[0,0,:]
    speed = res[1,0,:]
    accel = res[2,0,:]

    return (time,possi,speed,accel)
```

----

## Example
CODE:
```python=0
#Call function
y = traj_poly2(40,0,0,0,0,0,0,10,0.1)
plt.subplot(3,2,1)
plt.plot(y[0],y[1],'r')
plt.title('Position')
plt.subplot(3,2,3)
plt.plot(y[0],y[2],'g')
plt.title('Speed')
plt.subplot(3,2,5)
plt.plot(y[0],y[3],'b')
plt.title('Acceleration')

y = traj_poly2(40,0,0,0,0,0,10,20,0.1)
plt.subplot(3,2,2)
plt.plot(y[0],y[1],'r')
plt.title('Position delayed')
plt.subplot(3,2,4)
plt.plot(y[0],y[2],'g')
plt.title('Speed delayed')
plt.subplot(3,2,6)
plt.plot(y[0],y[3],'b')
plt.title('Acceleration delayed')
```

----

RESULT:

![](https://i.imgur.com/tgQfCLT.png)

---

# Linear segment with parabolic blends

![](https://i.imgur.com/Tmrg0No.png)

----

1. Because the previous movement method actuates our motor to full speed only at 50% of the total time
2. That case, we need more linear area to drive our motor full speed
3. Check Ref [[3]](https://smartech.gatech.edu/bitstream/handle/1853/41948/ParabolicBlends.pdf). This guy has a very intuitive explanation that helps me write the code
4. We use via a transition between linear movement
5. Around this via we introduce blend time to smooth the speed transition
6. On linear area speed is constant
7. On parabolic area acceleration is constant but speed is changing, for example  from $v_0$ to $0$ then to $v_f=-v_0$. 

----

## Linear phase
At the linear phase we use this equation
$$
q(t) = q_i + v_i*(t-T_i)\\
\dot{q}(t) = v_i\\
\ddot{q}(t) = 0
$$

Where $T_i$ is the time delay. $v_i$ can be calculated or specified
I have 2 different codes for different needs. If you only use linear you might want `line2()`. `line()` will be used in our next function

----

How to use it?
1. Specify 
    a. Initial position
    b. Speed
    c. Time initial
    d. Time final
2. Or specify
    a. Position initial
    b. Position final
    c. Time initial
    d. Time final

----

CODE:
```python=0
# function for linear interpolation
def line(point0, v0, t0, t1, step=1):
    # Generate a series of timestep
    t = np.arange(t0, t1+step,step)#makit one column
    # Calculate velocity
    v = v0
    #time shift
    Ti = t0
    #equation
    s = point0 + v*(t-Ti)
    v = np.ones(t.size)*v
    a = np.zeros(t.size)
    return (t,s,v,a)

# function for linear interpolation
def line2(point0, point1, t0, t1, step=1):
    # Generate a series of timestep
    t = np.arange(t0, t1+step,step)#makit one column
    # Calculate velocity
    v = (point1-point0)/(t1-t0)
    #time shift
    Ti = t0
    #equation
    s = point0 + v*(t-Ti)
    v = np.ones(t.size)*v
    a = np.zeros(t.size)
    return (t,s,v,a)

```

----

### Example 1
CODE:
```python=0
# %%
# Call function
y = line(0,1,-10,10,1)
plt.title('LINE')
plt.plot(y[0],y[1])
```
![](https://i.imgur.com/UMryohx.png)


----

### Example 2
CODE:
```python=0
#%%
# Call function
y = line2(10,20,-10,10,1)
plt.title('LINE2')
plt.plot(y[0],y[1])
```
![](https://i.imgur.com/cBfzRTX.png)


----

## Parabolic phase

This phase we use equation
$$
q(t) = q_i + v_{i-1}(t-T_i) + \frac{1}{2}a(t-T_i+\frac{t_i^b}{2})^2\\
\dot{q}(t) = v_{i-1}(t-T_i) + a(t-T_i+\frac{t_i^b}{2})\\
\ddot{q}(t) = a
$$

$q_i$ is the starting value. $i$ is the index of the via. $t$ is between $T_i-\frac{t_i^b}{2}<t<T_i+\frac{t_i^b}{2}$. $T_i$ is the time via happens and $t_i^b$ is the amount of blend *around* that via. So the parabolic phase start half $t_i^b$ before $T_i$ and half after that. Because the signal start at $T_i-\frac{t_i^b}{2}$ so we shift the equation to $T_i-\frac{t_i^b}{2}$.

----

CODE:
```python=0
def parab(p0, v0, v1, t0, t1, step=1):
    # Generate a series of timestep
    t = np.arange(t0, t1+step,step)
    #calculate acceleration
    a = (v1-v0)/(t1-t0)
    #time shift
    Ti=t0
    # equation
    s = p0  +v0*(t-Ti) +0.5*a*(t-Ti)**2
    v = v0 + a*(t-Ti)
    a = np.ones(t.size)*a
    return (t,s,v,a)
```

How to use it? Specify:
1. Initial pose
2. Speed before via
3. Speed after via
4. Start of period which is $T_i-\frac{t_i^b}{2}$
5. Final of period which is $T_i+\frac{t_i^b}{2}$

----

CODE:
```python=0
#%%
y = parab(0, 5, 0, 0, 100, step=1)
plt.plot(y[0],y[1])
y = parab(y[1][-1], 0, -5, 100, 200, step=1)
plt.plot(y[0],y[1])
y = parab(50, 5, -5, 50, 250, step=1)
plt.plot(y[0],y[1])
```
RESULT:

![](https://i.imgur.com/mPVhTIp.png)

----

## Linear and parabolic phase combined

CODE:
```python=0
# %%
def lspb(via,dur,tb):
    #1. It must start and end at the first and last waypoint respectively with zero velocity
    #2. Note that during the linear phase acceleration is zero, velocity is constant and position is linear in time
    # if acc.min < 0 :
    #     print('acc must bigger than 0')
    #     return 0
    if ((via.size-1) != dur.size):
        print('duration must equal to number of segment which is via-1')
        return 0
    if (via.size <2):
        print('minimum of via is 2')
        return 0
    if (via.size != (tb.size)):
        print('acc must equal to number of via')
        return 0
    
    #=====CALCULATE-VELOCITY-EACH-SEGMENT=====
    v_seg=np.zeros(dur.size)
    for i in range(0,len(via)-1):
        v_seg[i]=(via[i+1]-via[i])/dur[i]

    #=====CALCULATE-ACCELERATION-EACH-VIA=====
    a_via=np.zeros(via.size)
    a_via[0]=(v_seg[0]-0)/tb[0]
    for i in range(1,len(via)-1):
        a_via[i]=(v_seg[i]-v_seg[i-1])/tb[i]
    a_via[-1]=(0-v_seg[-1])/tb[-1]

    #=====CALCULATE-TIMING-EACH-VIA=====
    T_via=np.zeros(via.size)
    T_via[0]=0.5*tb[0]
    for i in range(1,len(via)-1):
        T_via[i]=T_via[i-1]+dur[i-1]
    T_via[-1]=T_via[-2]+dur[-1]

    #=====GENERATING-CHART/GRAPH/FIGURE=====
    # q(t) = q_i + v_{i-1}(t-T_i) + \frac{1}{2}a(t-T_i+\frac{t_i^b}{2})^2  #parabolic phase
    # q(t) = q_i + v_i*(t-T_i)                 #linear phase
    #parabolic
    t,s,v,a = parab(via[0], 0, v_seg[0], T_via[0]-0.5*tb[0], T_via[0]+0.5*tb[0], step=1)
    time    = t
    pos     = s
    speed   = v
    accel   = a
    
    for i in range(1,len(via)-1):
        # linear
        t,s,v,a = lerp(pos[-1],v_seg[i-1],T_via[i-1]+0.5*tb[i],T_via[i]-0.5*tb[i+1],0.01)
        time    = np.concatenate((time,t))
        pos     = np.concatenate((pos,s))
        speed   = np.concatenate((speed,v))
        accel   = np.concatenate((accel,a))

        #parabolic
        t,s,v,a = parab(pos[-1], v_seg[i-1], v_seg[i], T_via[i]-0.5*tb[i+1], T_via[i]+0.5*tb[i+1], 0.01)
        time    = np.concatenate((time,t))
        pos     = np.concatenate((pos,s))
        speed   = np.concatenate((speed,v))
        accel   = np.concatenate((accel,a))

    # linear
    t,s,v,a = lerp(pos[-1],v_seg[-1],T_via[-2]+0.5*tb[-2],T_via[-1]-0.5*tb[-1],0.01)
    time    = np.concatenate((time,t))
    pos     = np.concatenate((pos,s))
    speed   = np.concatenate((speed,v))
    accel   = np.concatenate((accel,a))

    #parabolic
    t,s,v,a = parab(pos[-1], v_seg[-1], 0, T_via[-1]-0.5*tb[-1],  T_via[-1]+0.5*tb[-1], 0.01)
    time    = np.concatenate((time,t))
    pos     = np.concatenate((pos,s))
    speed   = np.concatenate((speed,v))
    accel   = np.concatenate((accel,a))

    print('v seg = ',v_seg,
    '\na via = ',a_via,
    '\nT via = ',T_via,
    '\ntime = ',time,
    '\npos = ',pos)

    return(v_seg,a_via,T_via,time,pos,speed,accel)
```

----

How to use it? Specify:
1. All the via
2. All the duration between via
3. Blend time

In Ref [[1]](https://link.springer.com/book/10.1007%2F978-3-642-20144-8) [[2]](http://www.mech.sharif.ir/c/document_library/get_file?uuid=5a4bb247-1430-4e46-942c-d692dead831f&groupId=14040) they prefer acceleration on each via as input. But I follow [[3]](https://smartech.gatech.edu/bitstream/handle/1853/41948/ParabolicBlends.pdf) to use blend time as an input because it is more intuitive, easy to imagine and understand. Instead of duration between via, we could use specific time each via, it will be easier to code. But don't input $t_b=0$ that means infinite acceleration in that blend $a=\frac{(v_1-v_0)}{t_b}$

----

CODE:
```python=0
# %%
via = np.asarray([0,40,0,40,0])
dur = np.asarray([20,20,20,20])
tb = np.asarray([1,1,1,1,1])*5
res=lspb(via,dur,tb)

plt.plot(res[2],via,'*',res[3],res[4])
```
RESULT:

![](https://i.imgur.com/LVZffTD.png)

----

We can plot acceleration, speed, together. As you can see that the speed is zero at the via with constant accelereration
```python=0
# %%
plt.plot(res[2],np.zeros(via.size),'*')
plt.plot(res[3],res[5])
plt.plot(res[3],res[6])
```

![](https://i.imgur.com/GLwGNHx.png)


----

Let's try anoter
CODE:
```python=0
# %%
via = np.asarray([0,30,40,10,0])-20
dur = np.asarray([20,20,20,20])
tb = np.asarray([1,1,1,1,1])*5
res=lspb(via,dur,tb)

plt.plot(res[2],via,'*',res[3],res[4])
```

![](https://i.imgur.com/4a59pkN.png)


----
This thime the transition speed is not zero
```python=0
# %%
plt.plot(res[2],np.zeros(via.size),'*')
plt.plot(res[3],res[5])
plt.plot(res[3],res[6])
```

![](https://i.imgur.com/CU2yTOo.png)

---

# References
[[1] Robotics vision and control Professor Peter Corke p.44](https://link.springer.com/book/10.1007%2F978-3-642-20144-8)

[[2] Introduction to robotics John J. Craig p.207](http://www.mech.sharif.ir/c/document_library/get_file?uuid=5a4bb247-1430-4e46-942c-d692dead831f&groupId=14040)

[[3] Turning Paths Into Trajectories Using Parabolic Blends Tobias Kunz and Mike Stilman](https://smartech.gatech.edu/bitstream/handle/1853/41948/ParabolicBlends.pdf)