// Rebuild all README SVGs directly from benchmark-summary.json
// node scripts/render_readme_charts.mjs
import {readFileSync,writeFileSync,mkdirSync} from "node:fs";
import {dirname} from "node:path";
const makeVisuals = function makeVisuals(r){
const c={bg:"#0B1426",panel:"#12223A",stroke:"#283B54",grid:"#314760",white:"#F3F8FF",muted:"#9AB0C7",green:"#61E7C6",blue:"#86A8F9",gray:"#6C819A",gold:"#F3CA85"};
const box=(x,y,w,h,fill,round=0,stroke="none")=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${round}" fill="${fill}" stroke="${stroke}"/>`;
const txt=(x,y,value,size=18,color=c.white,weight=500,anchor="start")=>`<text x="${x}" y="${y}" text-anchor="${anchor}" fill="${color}" font-size="${size}" font-family="Inter,Arial,sans-serif" font-weight="${weight}">${String(value).replaceAll("&","&amp;").replaceAll("<","&lt;")}</text>`;
const ln=(x1,y1,x2,y2,color=c.grid,width=1,opacity=1)=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" opacity="${opacity}"/>`;
const circ=(x,y,r,color)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${color}"/>`;
const wrap=(h,title,desc,body)=>`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 ${h}" role="img" aria-label="${title}"><title>${title}</title><desc>${desc}</desc><defs><linearGradient id="gradient" x1="0" x2="1"><stop stop-color="#238FBB"/><stop offset="1" stop-color="#61E7C6"/></linearGradient></defs>${box(0,0,1200,h,c.bg)}${body}</svg>`;
const heading=(section,title,subtitle)=>txt(64,53,section,14,c.green,750)+txt(64,107,title,38,c.white,800)+txt(64,139,subtitle,17,c.muted,450);
const names=["zero","follow","lead"],colors={zero:c.gray,follow:c.blue,lead:c.green},m=r.main,ratio=(m.lead.mean/m.zero.mean);
let s=box(25,22,1150,321,c.panel,20,c.stroke);
for(let x=690;x<=1140;x+=42)s+=ln(x,49,x,318,c.grid,1,.24);
for(let y=50;y<=320;y+=42)s+=ln(678,y,1145,y,c.grid,1,.24);
s+=txt(66,72,"EXPERIMENT 05   /   DIRECT WEIGHT PROGRAMMING",15,c.green,750);
s+=txt(63,153,"WEIGHT",72,c.white,820)+txt(63,228,"SMITHING",72,c.white,820);
s+=txt(68,272,"Programming neural behavior — no training loop.",18,c.muted,520);
s+=box(66,291,501,35,"#164843",8)+txt(83,316,`318 NONZERO WEIGHTS    •    ${ratio.toFixed(2)}× BASELINE REWARD`,15,c.green,730);
const layers=[[737,4],[838,6],[947,6],[1064,4]];
layers.forEach(([x,n],k)=>{
 if(k<3){let [nx,nn]=layers[k+1];for(let i=0;i<n;i++)for(let j=0;j<nn;j++)if((i+j)%2===0)s+=ln(x+8,105+i*154/(n-1),nx-8,105+j*154/(nn-1),(i+j)%4===0?c.green:c.grid,1,.37);}
 for(let i=0;i<n;i++){let yy=105+i*154/(n-1);s+=circ(x,yy,6,(i+k)%4===0?c.green:c.blue);}
 s+=txt(x,307,["INPUT","TANH","TANH","ACTION"][k],12,c.muted,600,"middle");
});
const hero=wrap(367,"Weight Smithing","Training-free direct neural weight editing on Breakout.",s);
s=heading("01  /  CONTROLLED TOY BENCHMARK","Five times the baseline reward","2,000 matched episode seeds · sticky action probability 0.10 · bootstrap 95% CIs");
s+=box(50,167,1100,334,c.panel,15,c.stroke);
const barX=320,barW=535;
for(let v=0;v<=10;v+=2){let x=barX+v/10*barW;s+=ln(x,203,x,466,c.grid,1,.7)+txt(x,489,String(v),13,c.muted,450,"middle");}
names.forEach((name,i)=>{
let o=m[name],y=238+i*97,v=o.mean,x=barX+v/10*barW;
s+=txt(78,y,["ZERO / RANDOM","FOLLOW / BALL POSITION","LEAD / BALL VELOCITY"][i],15,name==="lead"?c.green:c.muted,740);
s+=box(barX,y-27,x-barX,42,name==="lead"?"url(#gradient)":colors[name],7);
s+=txt(x+17,y+3,v.toFixed(2),23,c.white,770);
let lo=barX+o.ci95[0]/10*barW,hi=barX+o.ci95[1]/10*barW;
s+=ln(lo,y-6,hi,y-6,c.white,2)+ln(lo,y-13,lo,y+1,c.white,2)+ln(hi,y-13,hi,y+1,c.white,2);
});
s+=box(926,209,188,233,"#1A3249",12,c.stroke);
s+=txt(944,286,ratio.toFixed(2)+"×",51,c.green,820)+txt(945,318,"mean reward",17,c.muted,560);
s+=ln(945,341,1096,341,c.stroke);
s+=txt(945,391,"+"+m.lead.paired_delta.toFixed(2),32,c.gold,790)+txt(945,415,"paired score lift",15,c.muted,480);
s+=txt(60,532,"Independent toy Breakout, not the paper's MinAtar benchmark. Manually programmed weights.",14,c.muted,480);
const means=wrap(562,"Mean reward benchmark","Zero 1.576; follow 1.8815; lead 8.017 over 2000 episodes, with bootstrap confidence intervals.",s);
s=heading("02  /  THE SCORE DISTRIBUTION","Better than a lucky average","Empirical survival: probability that an episode scores at least the reward threshold");
s+=box(50,164,1100,393,c.panel,16,c.stroke);
const px=126,py=472,pw=663,ph=242;
for(let f=0;f<=1;f+=.25){let y=py-f*ph;s+=ln(px,y,px+pw,y,c.grid,1,.8)+txt(px-18,y+4,Math.round(f*100)+"%",13,c.muted,450,"end");}
for(let t=0;t<=40;t+=10){let x=px+pw*t/40;s+=ln(x,231,x,py,c.grid,1,.4)+txt(x,py+29,String(t),14,c.muted,500,"middle");}
names.forEach((name,i)=>{
const a=m[name].hist;let points=Array.from({length:41},(_,t)=>{let remaining=a.reduce((sum,v,j)=>sum+(j>=t?v:0),0)/m.episodes;return (px+t*pw/40).toFixed(1)+","+(py-remaining*ph).toFixed(1);});
s+=`<polyline points="${points.join(" ")}" fill="none" stroke="${colors[name]}" stroke-width="${name==="lead"?4:2.5}" stroke-linejoin="round"/>`;
s+=ln(154+i*160,197,183+i*160,197,colors[name],4)+txt(191+i*160,202,name.toUpperCase(),14,c.muted,700);
});
s+=txt(461,528,"Episode reward threshold",15,c.muted,500,"middle");
s+=box(834,215,270,289,"#1A3249",12,c.stroke)+txt(857,253,"SCORED AT LEAST FIVE",13,c.muted,720);
s+=txt(856,310,(m.lead.p_ge5*100).toFixed(1)+"%",49,c.green,820)+txt(856,341,"lead policy",16,c.muted);
s+=ln(855,359,1085,359,c.stroke);
s+=txt(856,398,"90TH PERCENTILE",14,c.muted,720)+txt(856,437,"23  vs.  3",29,c.white,800);
s+=txt(856,474,"LEAD vs. RANDOM",13,c.muted,670);
const dist=wrap(586,"Toy reward survival curves","Lead controller scores 5+ reward in 49.15% of episodes; random baseline 0.55%.",s);
s=heading("03  /  ROBUSTNESS","The cost of unreliable actions","Expected reward as the environment repeats actions with increasing probability");
s+=box(50,164,1100,395,c.panel,16,c.stroke);
const ax=135,ay=476,aw=678,ah=261;
for(let val=0;val<=16;val+=4){let y=ay-val/16*ah;s+=ln(ax,y,ax+aw,y,c.grid,1,.75)+txt(ax-20,y+4,String(val),13,c.muted,500,"end");}
r.sweep.forEach(z=>{let x=ax+z.sticky*aw/.5;s+=ln(x,218,x,ay,c.grid,1,.38)+txt(x,ay+28,Math.round(z.sticky*100)+"%",14,c.muted,500,"middle");});
names.forEach(name=>{
let points=r.sweep.map(z=>[ax+z.sticky*aw/.5,ay-z[name][0]/16*ah]);
s+=`<polyline points="${points.map(z=>z.join(",")).join(" ")}" fill="none" stroke="${colors[name]}" stroke-width="${name==="lead"?4:2.5}" stroke-linejoin="round"/>`;
points.forEach(([x,y],i)=>{let z=r.sweep[i][name];s+=ln(x,ay-z[1]/16*ah,x,ay-z[2]/16*ah,colors[name],2,.8)+circ(x,y,name==="lead"?6:4,colors[name]);});
});
s+=txt(475,534,"Sticky-action probability",15,c.muted,500,"middle");
s+=box(845,218,270,300,"#1A3249",12,c.stroke);
s+=txt(868,259,"LEAD @ 0% STICKY",14,c.muted,720)+txt(868,311,"14.67",45,c.green,810);
s+=ln(868,332,1087,332,c.stroke)+txt(868,370,"LEAD @ 50% STICKY",14,c.muted,720)+txt(868,420,"2.25",43,c.white,810);
s+=txt(868,472,"Still ahead of random —",15,c.gold,630)+txt(868,495,"but the gap shrinks.",15,c.muted,500);
const robust=wrap(590,"Reward versus sticky action probability","Lead reward declines 14.666, 8.017, 4.242 and 2.25 as action persistence increases.",s);
s=heading("04  /  INTERPRETABILITY","A sparse policy, not a trained black box","First-layer weight maps encode ball and paddle location as horizontal coordinates");
s+=box(50,164,1100,307,c.panel,16,c.stroke);
for(let group=0;group<3;group++){let x0=90+group*230;
s+=txt(x0,215,["CURRENT BALL","PAST BALL","PADDLE"][group],14,group===1?c.blue:c.green,710);
for(let y=0;y<10;y++)for(let x=0;x<10;x++)s+=box(x0+x*13.7,235+y*13.7,11.4,11.4,group===2&&y<9?"#203249":x<5?c.blue:c.green,2);
s+=txt(x0,399,["position signal","velocity signal","bottom-row position"][group],15,c.muted,450);
}
s+=box(781,210,318,218,"#1A3249",12,c.stroke);
s+=txt(806,250,"NONZERO PARAMETERS",14,c.muted,720)+txt(803,317,"1.55%",61,c.green,830);
s+=box(806,338,266,13,c.grid,7)+box(806,338,266*r.weights.edited_fraction,13,c.green,7);
s+=txt(806,391,"318 of 20,486 weights",20,c.white,700);
s+=txt(65,509,"Diagram shows input-coordinate patterns, not learned feature activations.",14,c.muted);
const sparse=wrap(538,"Sparse programmed weight map","Only 318 out of 20486 parameters are nonzero in the lead policy.",s);
return {"assets/hero.svg":hero,"assets/mean-reward.svg":means,"assets/reward-tail.svg":dist,"assets/robustness.svg":robust,"assets/weight-map.svg":sparse};
};
const data=JSON.parse(readFileSync("results/benchmark-summary.json","utf8"));
for (const [path,svg] of Object.entries(makeVisuals(data))){mkdirSync(dirname(path),{recursive:true});writeFileSync(path,svg+"\n");console.log(path);}
