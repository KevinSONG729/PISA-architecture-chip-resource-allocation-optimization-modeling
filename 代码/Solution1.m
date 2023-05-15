clc,clear;
data = readmatrix('C:\Users\Desktop\研赛\D\attachment1.csv');
data2 = readmatrix('C:\Users\Desktop\研赛\D\YilaiMatrix.csv');
% data3 = readmatrix();
% data2 = zeros(607, 607);
data3 = zeros(607, 607);
% data2(1,2) = 1;
% data2(1,3) = 2;
TCAM = data(:,2);
HASH = data(:,3);
ALU = data(:,4);
QUALIFY = data(:,5);
prob = optimproblem("ObjectiveSense", "max");
flowNum =90;
miu = 1e-6;
M = 1e6;
RD = repmat(sort(randperm(flowNum)), 607, 1);
x = optimvar('x', 607, flowNum, 'Type', 'integer', 'LowerBound', 0, 'UpperBound', 1);
y = optimvar('y', 1, flowNum, 'Type', 'integer', 'LowerBound', 0, 'UpperBound', 1);
z = optimvar('z', 607, 1, 'Type','integer','LowerBound',1,'UpperBound',flowNum);
con1 = [];
con2 = [];
for j = 1:607
    for i = j:607
        if(data2(i,j)==1) % i要在j前面
%             con1 = [con1; sum(x(i,:).*RD(i,:))<=(sum(x(j,:).*RD(j,:))-1)];
              con1 = [con1; z(i,:)<=z(j,:)-1];
        elseif(data2(i,j)==2) % i要在j前面或者相同位置
%             con2 = [con2; sum(x(i,:).*RD(i,:))<=sum(x(j,:).*RD(j,:))];
              con2 = [con2; z(i,:)<=z(j,:)];
        end
    end
end

con2
con1;
con3 = [];
for i = 1:607
    con3 = [con3; sum(x(i,:).*RD(i,:))==z(i,:)];
end

con4 = [sum(x, 2) == 1];
con5 = [
    sum(x.*repmat(TCAM, 1, flowNum)) <= 1;
    sum(x.*repmat(HASH, 1, flowNum)) <= 2;
    sum(x.*repmat(ALU, 1, flowNum)) <= 56;
    sum(x.*repmat(QUALIFY, 1, flowNum)) <= 64;
    
];
con6 = [
    miu*y <= sum(x.*repmat(TCAM, 1, flowNum));
    sum(x.*repmat(TCAM, 1, flowNum)) <= M*y;
];
con7 = sum(y(1:2:flowNum)) <= 5;
con8 = [];
if(flowNum>=17 && flowNum<=32)
    for idx = flowNum:-1:17
        con8 = [con8; sum(x(:,idx).*TCAM)+sum(x(:,idx-16).*TCAM)<=1];
        con8 = [con8; sum(x(:,idx).*HASH)+sum(x(:,idx-16).*HASH)<=3];
    end
elseif(flowNum>32)
    for idx = 32:-1:17
        con8 = [con8; sum(x(:,idx).*TCAM)+sum(x(:,idx-16).*TCAM)<=1];
        con8 = [con8; sum(x(:,idx).*HASH)+sum(x(:,idx-16).*HASH)<=3];
    end
end

con9 = [];
for i = 1:607
    for j = i:607
        if(data3(i,j)==1 && z(i,:)==z(j,:)) % i与j在同一条路径上且位于同一流水线
            con9 = [con9; ];
        end
    end
end

prob.Objective = sum(x.*(repmat(122/123* TCAM+121/123* HASH+67/123*ALU+59/123*QUALIFY, 1, flowNum)), "all") / (flowNum*(1+2+56+64));
% prob.Objective = sum(x.*(repmat(TCAM+HASH+ALU+QUALIFY, 1, flowNum)), "all") / (flowNum*(1+2+56+64));
prob.Constraints.con1 = con1; % 写后读/写后写/控制依赖约束
prob.Constraints.con2 = con2; % 读后写依赖约束
prob.Constraints.con3 = con3; % x与z绑定
prob.Constraints.con4 = con4; % 指派约束
prob.Constraints.con5 = con5; % 资源约束
prob.Constraints.con6 = con6; % 额外约束1（偶数级有TACM的数目不大于5）
prob.Constraints.con7 = con7; % 额外约束1
if(flowNum>=17)
    prob.Constraints.con8 = con8; % 额外约束2（折叠级约束）
end

sol = prob.solve();

% 目前差的工作:把依赖关系作为data2传入脚本，其中值为1代表硬依赖，值为2代表软依赖。(已完成)
% 问题2还会增加新的约束，预计要把data3(607*607，两基本块是否在一条路径上)传入脚本
% 
sol.z
