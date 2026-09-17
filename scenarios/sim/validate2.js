for(const REALv of [0,1]){process.env.REAL=String(REALv);delete require.cache[require.resolve('./ptr_real.js')];const Q=require('./ptr_real.js');
  const orb=[],orbObs=[],lin=[];let laps=0,spinFrames=0,frames=0;
  for(let seed=1;seed<=5;seed++){
    let W=Q.World(0,seed,'orbit');W.balls=[W.balls[0]];const b=W.balls[0];let prev=null,sw=0;
    while(W.t<60){Q.step(W);frames++;if(b.spinning)spinFrames++;
      const a=Math.atan2(b.p[1]-55,b.p[0]-70);if(prev!==null){let d=a-prev;if(d>Math.PI)d-=2*Math.PI;if(d<-Math.PI)d+=2*Math.PI;sw+=d}prev=a;
      if(b.state==='go'&&W.t>10){orb.push(Math.hypot(b.p[0]-70,b.p[1]-55)-30);const o=b.lastObs||{p:b.p};orbObs.push(Math.hypot(o.p[0]-70,o.p[1]-55)-30)}}
    laps+=Math.abs(sw)/(2*Math.PI);
    let W2=Q.World(0,seed,'cross');W2.balls=[W2.balls[0]];const b2=W2.balls[0];
    while(W2.t<30&&b2.state==='go'){Q.step(W2);if(b2.path&&b2.s>15&&b2.s<b2.path.total-15){const o=b2.lastObs||{p:b2.p};lin.push(o.p[1]-55)}}}
  const rms=a=>Math.sqrt(a.reduce((x,y)=>x+y*y,0)/Math.max(1,a.length));
  console.log((REALv?'REALISTIC':'CLEAN    ')+` laps/60s ${(laps/5).toFixed(2)} spinning ${(100*spinFrames/frames).toFixed(0)}% | orbit rms true ${rms(orb).toFixed(2)} camera ${rms(orbObs).toFixed(2)} (real 2.5) | line rms camera ${rms(lin).toFixed(2)} (real 1.4)`)}
