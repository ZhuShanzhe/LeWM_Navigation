"""Small topology stress environment. New conservative collision semantics are explicit."""
import numpy as np, torch
from stable_worldmodel.envs.two_room.env import TwoRoomEnv

def topology_specs():
    return [
        make_spec('bridge_two_room', [14,112,210], [14,210], [(0,1)], door_overrides={(0,1):49}),
        make_spec('four_room_chain', [14,112,210], [14,112,210], [(0,1),(0,2),(2,3)]),
        make_spec('four_room_cycle', [14,112,210], [14,112,210], [(0,1),(0,2),(2,3),(1,3)]),
        make_spec('six_room_branch', [14,79,145,210], [14,112,210], [(0,1),(1,2),(1,4),(3,4),(4,5)]),
    ]

def make_spec(name,xs,ys,edges,door_overrides=None):
    cols=len(xs)-1; rows=len(ys)-1
    edges={tuple(sorted(e)) for e in edges};over=door_overrides or {};rects=[];gates=[]
    rooms=[[(xs[c]+xs[c+1])/2,(ys[r]+ys[r+1])/2] for r in range(rows) for c in range(cols)]
    bounds=[[xs[c],xs[c+1],ys[r],ys[r+1]] for r in range(rows) for c in range(cols)]
    for r in range(rows):
        for c in range(cols-1):
            edge=(r*cols+c,r*cols+c+1);x=xs[c+1];lo=ys[r];hi=ys[r+1]
            if edge in edges:
                d=over.get(edge,(lo+hi)/2);rects.extend([[x-5,x+5,lo,d-14],[x-5,x+5,d+14,hi]])
                gates.append({'edge':list(edge),'xy':[x,d],'normal':[1,0]})
            else:rects.append([x-5,x+5,lo,hi])
    for r in range(rows-1):
        for c in range(cols):
            edge=(r*cols+c,(r+1)*cols+c);y=ys[r+1];lo=xs[c];hi=xs[c+1]
            if edge in edges:
                d=over.get(edge,(lo+hi)/2);rects.extend([[lo,d-14,y-5,y+5],[d+14,hi,y-5,y+5]])
                gates.append({'edge':list(edge),'xy':[d,y],'normal':[0,1]})
            else:rects.append([lo,hi,y-5,y+5])
    return {'id':name,'rectangles':rects,'room_centers':rooms,'room_bounds':bounds,'edges':[list(e) for e in sorted(edges)],'gates':gates,'scope':'full-observation custom topology diagnostic; not official TwoRoom'}

class TopologyEnv(TwoRoomEnv):
    def __init__(self,spec,**kwargs):
        self.topology_spec=spec
        self.rects=np.asarray(spec['rectangles'],dtype=np.float64).reshape(-1,4)
        self.effective=self.rects+np.array([-7,7,-7,7])
        super().__init__(**kwargs)
        self.env_name='TopologyDiagnostic'
        self._mask=None

    def valid(self,p):
        p=np.asarray(p)
        if np.any(p<21) or np.any(p>203):return False
        q=self.effective
        return not np.any((p[0]>=q[:,0])&(p[0]<=q[:,1])&(p[1]>=q[:,2])&(p[1]<=q[:,3]))

    def clear(self,a,b):
        a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
        if not self.valid(a) or not self.valid(b):return False
        d=b-a
        for x0,x1,y0,y1 in self.effective:
            enter=0.;leave=1.
            for k,(lo,hi) in enumerate([(x0,x1),(y0,y1)]):
                if abs(d[k])<1e-12:
                    if a[k]<lo or a[k]>hi:enter=2.;break
                else:
                    t0=(lo-a[k])/d[k];t1=(hi-a[k])/d[k]
                    enter=max(enter,min(t0,t1));leave=min(leave,max(t0,t1))
            if enter<=leave:return False
        return True

    def _constrain_agent_not_in_wall(self,p):
        return self.valid(p)

    def reset(self,seed=None,options=None):
        opts=dict(options or {})
        opts.setdefault('state',np.array(self.topology_spec['room_centers'][0],np.float32))
        opts.setdefault('target_state',np.array(self.topology_spec['room_centers'][-1],np.float32))
        assert self.valid(opts['state']) and self.valid(opts['target_state'])
        return super().reset(seed=seed,options=opts)

    def _get_obs(self):
        out=torch.zeros(10,dtype=torch.float32)
        out[:2]=self.agent_position;out[2:4]=self.target_position
        return out

    def _wall_and_door_masks(self):
        if self._mask is None:
            x=self.grid_x;y=self.grid_y
            mask=torch.zeros((224,224),dtype=torch.bool)
            for x0,x1,y0,y1 in self.rects:mask|=(x>=x0)&(x<=x1)&(y>=y0)&(y<=y1)
            mask[:,10:14]=True;mask[:,210:214]=True;mask[10:14,:]=True;mask[210:214,:]=True
            self._mask=mask
        return self._mask,torch.zeros_like(self._mask)

    def _apply_collisions(self,pos1,pos2):
        a=pos1.numpy().astype(float);b=np.clip(pos2.numpy().astype(float),21,203)
        assert self.valid(a),a
        if self.clear(a,b):return torch.tensor(b,dtype=torch.float32)
        delta=b-a;n=max(1,int(np.ceil(np.linalg.norm(delta)/.5)));inc=delta/n
        for _ in range(n):
            candidates=[a+inc,a+np.array([inc[0],0]),a+np.array([0,inc[1]])]
            for q in candidates:
                q=np.asarray(q,dtype=np.float32).astype(float)
                if self.clear(a,q):a=q;break
        assert self.valid(a)
        return torch.tensor(a,dtype=torch.float32)

    def _set_state(self,p):
        assert self.valid(p),p
        super()._set_state(p)

    def _set_goal_state(self,p):
        assert self.valid(p),p
        super()._set_goal_state(p)
