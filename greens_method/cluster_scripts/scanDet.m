(* ::Package:: *)

cPath = FileNameTake[$InputFileName,{1,-2}];
If[$OperatingSystem=="Windows",
	Print["Windows detected, adjusting path"];
	cPath= StringReplace[cPath,{"\\"->"\\\\"}];]
SetDirectory[cPath];
Print["Setting directory to: "];
Print[Directory[]];

Print["Reading arguments from command line"];
ellMin=ToExpression[$CommandLine[[-1]]];
(* ellMax=ToExpression[$CommandLine[[-1]]]; *)



VV = beta*H^2 *vev^2(1/(beta* epsilon^2)- phi^2/2-b/3*(phi)^3+1/4*(phi)^4);
vPrime =D[VV,phi];
rules = {beta->45, b->25/100, epsilon->46/1000,H->1};
tol = 10^(-15);
chiMin=1/10000;
chiMax = Pi-chiMin;
numpoints=200;
MyHeaviside[x_]=Piecewise[{{0,x<0},{1/2,x==0},{1,x>0}}];
bounceLoad[chi_]=Get["bounceJuan20DigitsExtended.wdx"];
m2\[Phi]Eff[bounce_,chi_]:=FullSimplify[(D[VV,{phi,2}]/vev^2)/H^2/.Join[rules,{phi->bounceLoad[chi]}]];
m2\[Phi]EffInterp := Interpolation[Table[{chi,m2\[Phi]Eff[bounceLoad,chi]},{chi,chiMin,Pi,(Pi-chiMin)/400}],Method->"Hermite",InterpolationOrder->1];
Print["Tolerance, boundaries and potential defined"];


Op3[f_,ell_,chi_,s_]:=(-3Cot[chi] D[f,chi]-D[D[f,chi],chi])+Sin[chi]^(-2)ell(ell+2)f+ m2\[Phi]EffInterp[chi]f+s f;
eq3[ell_,chi_,s_]= Op3[G[chi],ell,chi,s];
sol[ell_,s_,pm_,equation_]:=If[pm==-1,NDSolve[{equation[ell,chi,s]==0,G[chiMin]==0,G'[chiMin]==1},G,{chi,chiMin,chiMax},WorkingPrecision->20,PrecisionGoal->10,MaxSteps->100000,InterpolationOrder->All,MaxStepSize->2*(chiMax-chiMin)/(4numpoints)],
NDSolve[{equation[ell,chi,s]==0,G[chiMax]==0,G'[chiMax]==-1},G,{chi,chiMin,chiMax},WorkingPrecision->20,PrecisionGoal->10,MaxSteps->100000,InterpolationOrder->All,MaxStepSize->2*(chiMax-chiMin)/(4numpoints)]];
f1less[eq_,ell_,chi_,s_]:=G[chi]/.sol[ell,s,-1,eq][[1]];
f1great[eq_,ell_,chi_,s_]:=G[chi]/.sol[ell,s,1,eq][[1]];

wronsk1[chi_,flessTemp_ ,fgreatTemp_]:=-flessTemp[chi]*D[fgreatTemp[up],up] +fgreatTemp[chi]*D[flessTemp[up],up]/.{up->chi};
greenScalar[eq_,ell_,s_,chi_,chiP_]:= Module[{f1lessEval,f1greatEval,wronskEval},
	f1lessEval[pt_] = f1less[eq,ell,chi,s]/.{chi->pt};
	f1greatEval[pt_] = f1great[eq,ell,chi,s]/.{chi->pt};
	wronskEval[pt_] =wronsk1[chi,f1lessEval,f1greatEval]/.{chi->pt};
	Return[(MyHeaviside[chiP-chi]*f1lessEval[chi]*f1greatEval[chiP]+MyHeaviside[chi-chiP]*f1greatEval[chi]*f1lessEval[chiP])*(H^(3))/(wronskEval[chiP])/.rules]
	];
Print["Differential equation solver defined"];

reconstructed={};
Print[StringJoin["Starting scan for ell=",ToString[ellMin]]];

tt = AbsoluteTiming[
$DateStringFormat="DateTime";
Print[StringJoin["Proccess started on: ", DateString[]]];
(* For[ell=ellMin,ell<ellMax,ell++, *)
For[ss=0,ss<10001,ss=ss+15,
currentGreen[chi_,chiP_] = greenScalar[eq3,ellMin,ss,chi,chiP];
reconstructed=Join[reconstructed,Table[{N[chiVal,10],ellMin,ss,N[currentGreen[chiVal,chiVal],15]},{chiVal,chiMin,chiMax,(chiMax-chiMin)/500}]];
If[Mod[ss,150]==0,Print[StringJoin["Computed s=",ToString[ss]]]];
];
Export[StringJoin["greenSparam-",ToString[ellMin],".csv"],reconstructed] (* ;]; *)
];
Print[StringJoin["Ellapsed time (min): ", ToString[N[tt[[1]]/60]]]];
Exit[];




