data=load('iris.txt');
x=data(:,1:4);
theta1=zeros(3, size(x,2)+1);
m=size(x,1);
y=zeros(size(x,1),1);
y(1:50)=1;
y(50:100)=2;
y(100:150)=3;
%training-set
xtr=[x(1:30,:);x(50:80,:);x(100:130,:)];
ytr=[y(1:30);y(50:80);y(100:130)];
%test-set
xt=[x(30:50,:);x(80:100,:);x(130:150,:)];
yt=[y(30:50);y(80:100);y(130:150)];
xtr=[ones(size(xtr,1),1) xtr]; 
xt=[ones(size(xt,1),1) xt]; 
%================================================
% Calculating the parameters theta1
%================================================
for j=1:3
c=(ytr==j);
alpha=0.03;
for i=1:1000
  k=xtr*theta1';
  sig=(1./(1+(exp(-k))));
  l=xtr'*((sig)-c);
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
for i=1:size(xt,1)
  max=0;
  pos=0;
  for j=1:3
   d=(1/(1+exp(-theta1(j,:)*xt(i,:)')));
   if(d>max)
     max=d;
     pos=j;
   endif
   pred(i)=pos;
  endfor   
endfor
acc=0;
for i=1:size(xt,1)
  if(pred(i)==yt(i))
    acc=acc+1;
  endif
endfor  
acc1=(acc/size(xt,1))*100;