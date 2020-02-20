data=load('iris.txt');
x=data(:,1:4);
theta1=zeros(3, size(x,2)+1);
m=size(x,1);
x=[ones(m,1) x];
y=zeros(size(x,1),1);
y(1:50)=1;
y(50:100)=2;
y(100:150)=3;
%================================================
% Calculating the parameters theta1
%================================================
for j=1:3
c=(y==j);
alpha=0.03;
for i=1:1000
  k=x*theta1';
  sig=(1./(1+(exp(-k))));
  l=x'*((sig)-c);
  temp1=theta1(j,1)-((alpha/m)*l(1,j));
  temp2=theta1(j,2)-((alpha/m)*l(2,j));
  temp3=theta1(j,3)-((alpha/m)*l(3,j));
  temp4=theta1(j,4)-((alpha/m)*l(4,j));
  temp5=theta1(j,5)-((alpha/m)*l(5,j));
  theta1(j,1)=temp1;
  theta1(j,2)=temp2;
  theta1(j,3)=temp3;
  theta1(j,4)=temp4;
  theta1(j,5)=temp5;
  u=(c'*log(sig))+((1.-c)'*log(1.-(sig)));
  J(i)=(-1/m)*sum(u);
  hold on;
  plot(i, J(i),"linestyle",'--',"marker",'.');
endfor
%=====================================================
%Calculating the accuracy on training set
%===================================================== 
endfor
for i=1:150
  max=0;
  pos=0;
  for j=1:3
   d=(1/(1+exp(-theta1(j,:)*x(i,:)')));
   if(d>max)
     max=d;
     pos=j;
   endif
   pred(i)=pos;
  endfor   
endfor
acc=0;
for i=1:m
  if(pred(i)==y(i))
    acc=acc+1;
  endif
endfor  
acc1=(acc/150)*100;