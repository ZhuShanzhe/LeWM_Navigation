"""Privileged geometry/pose waypoint diagnostic, never a deployable visual method."""
from collections import deque
import numpy as np
from topology_env import TopologyEnv

class OracleWaypointRouter:
    def __init__(self,spec,cases,mode='waypoint',threshold=8.):
        assert mode in ['waypoint','final']
        self.mode=mode;self.threshold=threshold;self.cases=cases;self.spec=spec
        self.routes=[];self.indices=np.zeros(len(cases),dtype=int);self.events=[];self.calls=0;self.goal_replacements=0
        self.cache={};env=TopologyEnv(spec);env.reset(seed=9104401)
        adj={i:[] for i in range(len(spec['room_centers']))}
        for a,b in spec['edges']:adj[a].append(b);adj[b].append(a)
        for case in cases:
            a=case['start_room'];b=case['goal_room'];todo=deque([[a]]);seen={a}
            while todo:
                rooms=todo.popleft()
                if rooms[-1]==b:break
                for z in sorted(adj[rooms[-1]]):
                    if z not in seen:seen.add(z);todo.append(rooms+[z])
            assert rooms[-1]==b
            route=[]
            if mode=='waypoint':
                for u,v in zip(rooms,rooms[1:]):
                    gate=next(g for g in spec['gates'] if set(g['edge'])=={u,v})
                    g=np.array(gate['xy'],dtype=np.float32)
                    direction=np.array(spec['room_centers'][v])-np.array(spec['room_centers'][u]);direction/=np.linalg.norm(direction)
                    route.extend([g-20*direction,g+20*direction])
            route.append(np.array(case['goals']['75']['xy'],dtype=np.float32))
            start=np.array(case['start_xy'])
            for target in route:
                assert env.valid(target) and (mode=='final' or env.clear(start,target)),(spec['id'],case['episode'],start,target)
                key=tuple(np.asarray(target,dtype=np.float32))
                if key not in self.cache:
                    env._set_state(np.array(key,dtype=np.float32));self.cache[key]=env.render().copy()
                start=target
            self.routes.append([np.array(x,dtype=np.float32) for x in route])
        env.close()
    def transform(self,info):
        # Raw observations copied; real environment final goal and terminal conditions untouched.
        n=len(self.cases);out=dict(info);goals=np.asarray(info['goal']).copy()
        assert goals.shape[0]==n and goals.shape[-3:]==(224,224,3)
        positions=np.asarray(info['proprio']).reshape(n,-1,2)[:,-1]
        dead=np.asarray(info.get('terminated',np.zeros(n)),dtype=bool).reshape(n,-1)[:,-1]
        for i,(route,pos) in enumerate(zip(self.routes,positions)):
            if dead[i]:continue
            before=int(self.indices[i])
            while self.indices[i]<len(route)-1 and np.linalg.norm(pos-route[self.indices[i]])<=self.threshold:
                self.indices[i]+=1
            if before!=self.indices[i]:self.events.append({'call':self.calls,'episode':self.cases[i]['episode'],'before':before,'after':int(self.indices[i])})
            key=tuple(route[self.indices[i]])
            goals[i]=self.cache[key]
            self.goal_replacements+=1
        out['goal']=goals;self.calls+=1
        return out
    def report(self):
        return {'privileged':True,'extra_information':'true room graph, doorway coordinates and exact current pose; synthetic waypoint goal images',
                'mode':self.mode,'threshold_px':self.threshold,'calls':self.calls,'goal_replacements':self.goal_replacements,'cached_goal_images':len(self.cache),
                'buffer_flushing':False,'final_environment_goal_unchanged':True,'events':self.events,
                'cases':[{'episode':c['episode'],'route':[x.tolist() for x in route],'index':int(i),'intermediate_reached':int(i),'intermediate_total':len(route)-1} for c,route,i in zip(self.cases,self.routes,self.indices)],
                'limits':'Oracle scaffold diagnostic, not learned hierarchy or deployment algorithm; threshold/progress uses true state.'}
