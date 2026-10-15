"""SuperPoint/LightGlue + online RGB-D frontier navigation, an adapted baseline.

This is NOT a reproduction of IEVE/UniGoal/Mod-IIN. It instantiates their
modular correspondence/exploration pattern with published pretrained features.
Inputs: image goal, RGB-D, perfect relative odometry and task type. No navmesh,
semantic mask, scene identifier, instance ID, target pose or target depth.
"""
import hashlib, heapq, json, math, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
ROOT=Path('/root/autodl-tmp/jepa_vln_baselines_20261015')
sys.path.insert(0,str(ROOT/'repos/LightGlue'))

def load_publisher_state(url,*args,**kwargs):
    path=ROOT/'weights'/url.split('/')[-1].split('?')[0]
    if not path.exists():raise FileNotFoundError(path)
    return torch.load(path,map_location='cpu',weights_only=True)

# Enforce local, restricted loading even in upstream model constructors.
torch.hub.load_state_dict_from_url=load_publisher_state
_load_state_dict=torch.nn.Module.load_state_dict
_load_reports=[]
def checked_load(self,state_dict,*args,**kwargs):
    result=_load_state_dict(self,state_dict,*args,**kwargs)
    _load_reports.append((self.__class__.__name__,list(result.missing_keys),list(result.unexpected_keys)))
    return result
torch.nn.Module.load_state_dict=checked_load
from lightglue import SuperPoint, LightGlue

class Modular:
    def __init__(self,device='cuda'):
        self.device=device
        self.extractor=SuperPoint(max_num_keypoints=512).eval().to(device)
        self.matcher=LightGlue(features='superpoint',depth_confidence=.95,width_confidence=.99).eval().to(device)
        # The upstream constructor first probes old names, then remaps them and
        # loads again. Audit the FINAL load, not that intermediate compatibility
        # probe. The threshold buffer is deterministically derived, not learned.
        final=next(r for r in reversed(_load_reports) if r[0]=='LightGlue')
        missing,unexpected=final[1:]
        if set(missing)-{'confidence_thresholds'}:raise RuntimeError('Missing learned matcher weights: '+str(missing))
        if any(not(k.startswith('log_assignment.') and k.endswith('.r')) for k in unexpected):
            raise RuntimeError('Unreviewed extra matcher weights: '+str(unexpected))
        self.weight_audit={'all_current_learned_parameters_loaded':True,'derived_buffers':missing,'unused_upstream_legacy_keys':unexpected}
        self.k=np.array([[128.,0,127.5],[0,128.,127.5],[0,0,1.]])

    def features(self,rgb):
        im=torch.from_numpy(rgb.copy()).permute(2,0,1).float().to(self.device)/255.
        return self.extractor.extract(im,resize=None)

    def reset(self,goal,task):
        with torch.inference_mode():self.goal=self.features(goal)
        self.task=task;self.grid=np.zeros((601,601),np.uint8);self.visits=np.zeros_like(self.grid,dtype=np.float32)
        self.path=[];self.t=0;self.confirm=0;self.last=None;self.stuck=0;self.pending=None;self.matched_goal=None;self.match_age=0

    def cell(self,x,z):return int(round(z/.1))+300,int(round(x/.1))+300
    def world(self,r,c):return (c-300)*.1,(r-300)*.1

    def update_map(self,depth,pose):
        x,z,yaw=pose;cy,cx=self.cell(x,z);s,c=math.sin(yaw),math.cos(yaw)
        for u in range(4,256,8):
            rows=np.arange(16,240,4);d=depth[rows,u];height=1.25-(rows-127.5)*d/128.
            valid=np.isfinite(d)&(d>.05)&(d<7.)
            obstacles=valid&(height>.12)&(height<1.65)
            if obstacles.any():dd=float(d[obstacles].min());blocked=True
            elif valid.any():dd=min(6.,float(d[valid].max()));blocked=False
            else:continue
            rx=(u-127.5)*dd/128.;rz=-dd
            ex=x+c*rx+s*rz;ez=z-s*rx+c*rz;er,ec=self.cell(ex,ez)
            if not(1<=er<600 and 1<=ec<600):continue
            cv2.line(self.grid,(cx,cy),(ec,er),1,1)
            if blocked:cv2.circle(self.grid,(ec,er),1,2,-1)
        # The robot footprint is known free, but never clear distant obstacles.
        cv2.circle(self.grid,(cx,cy),1,1,-1);self.visits[cy,cx]+=1
        occupied=cv2.dilate((self.grid==2).astype(np.uint8),np.ones((5,5),np.uint8))>0
        safe=(self.grid==1)&(~occupied)
        safe[max(0,cy-1):cy+2,max(0,cx-1):cx+2]=True
        return safe,(cy,cx)

    def astar(self,safe,start,goal):
        q=[(0.,0.,start)];seen={};parent={};best=start;bestdist=1e9
        while q and len(seen)<20000:
            _,cost,p=heapq.heappop(q)
            if p in seen:continue
            seen[p]=cost;dist=math.hypot(p[0]-goal[0],p[1]-goal[1])
            if dist<bestdist:best=p;bestdist=dist
            if dist<=2.:best=p;break
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr,nc=p[0]+dr,p[1]+dc
                if not(0<=nr<601 and 0<=nc<601) or not safe[nr,nc] or (nr,nc) in seen:continue
                if dr and dc and not(safe[nr,p[1]] and safe[p[0],nc]):continue
                ng=cost+math.hypot(dr,dc)
                old=parent.get((nr,nc))
                if old is not None and old[1]<=ng:continue
                parent[(nr,nc)]=(p,ng);heapq.heappush(q,(ng+math.hypot(nr-goal[0],nc-goal[1]),ng,(nr,nc)))
        path=[]
        while best!=start and best in parent:path.append(best);best=parent[best][0]
        return path[::-1]

    def match(self,rgb,depth,pose):
        with torch.inference_mode():
            current=self.features(rgb);m=self.matcher({'image0':self.goal,'image1':current})
        pairs=m['matches'][0].cpu().numpy()
        if len(pairs)<8:return None,{'matches':len(pairs),'inliers':0}
        g=self.goal['keypoints'][0,pairs[:,0]].cpu().numpy();p=current['keypoints'][0,pairs[:,1]].cpu().numpy()
        if self.task=='instance_image_nav':
            core=(g[:,0]>38)&(g[:,0]<218)&(g[:,1]>38)&(g[:,1]<218);g,p=g[core],p[core]
        if len(p)<8:return None,{'matches':len(p),'inliers':0}
        _,mask=cv2.findHomography(g,p,cv2.RANSAC,5.)
        good=mask[:,0].astype(bool) if mask is not None else np.zeros(len(p),bool)
        pp=p[good];gg=g[good]
        info={'matches':len(p),'inliers':int(good.sum())}
        if len(pp)<8:return None,info
        uv=np.rint(pp).astype(int);uv=np.clip(uv,0,255);ds=depth[uv[:,1],uv[:,0]]
        valid=np.isfinite(ds)&(ds>.05)&(ds<8.)
        pp,gg,ds=pp[valid],gg[valid],ds[valid]
        if len(ds)<8:return None,info
        if self.task=='place_image_nav':
            xyz=np.column_stack(((pp[:,0]-127.5)*ds/128.,(pp[:,1]-127.5)*ds/128.,ds)).astype(np.float32)
            ok,rv,tv,inliers=cv2.solvePnPRansac(xyz,gg.astype(np.float32),self.k,None,flags=cv2.SOLVEPNP_EPNP,
                                               iterationsCount=100,reprojectionError=5.)
            if not ok or inliers is None or len(inliers)<8:return None,info
            r,_=cv2.Rodrigues(rv);cam=(-r.T@tv).ravel();local=np.array([cam[0],-cam[2]])
            info['predicted_goal_distance_m']=float(np.linalg.norm(local))
        else:
            local=np.array([np.median((pp[:,0]-127.5)*ds/128.),-np.median(ds)])
            info['predicted_goal_distance_m']=float(np.percentile(ds,25))
        x,z,yaw=pose;s,c=math.sin(yaw),math.cos(yaw)
        world=np.array([x+c*local[0]+s*local[1],z-s*local[0]+c*local[1]])
        return world,info

    def act(self,rgb,depth,pose):
        self.t+=1;safe,start=self.update_map(depth,pose)
        x,z,yaw=pose
        if self.last is not None and np.linalg.norm(np.array([x,z])-self.last)<.03 and self.pending==1:self.stuck+=1
        else:self.stuck=0
        self.last=np.array([x,z]);info={}
        # Match every second decision; the map is updated at every decision.
        if self.t%2==1 or self.confirm:
            goal,info=self.match(rgb,depth,pose)
            if goal is not None:
                self.matched_goal=goal;self.match_age=0
                threshold=.65 if self.task=='place_image_nav' else 1.0
                if info['inliers']>=12 and info['predicted_goal_distance_m']<threshold:self.confirm+=1
                else:self.confirm=0
                if self.confirm>=2:return 0,info
            else:self.confirm=0;self.match_age+=1
        if self.stuck>=2:
            # Mark only the attempted occupied direction, not the whole map.
            r,c=self.cell(x-.3*math.sin(yaw),z-.3*math.cos(yaw))
            if 0<=r<601 and 0<=c<601:cv2.circle(self.grid,(c,r),1,2,-1)
            self.path=[];self.pending=2;return 2,info
        target=None
        if self.matched_goal is not None and self.match_age<15:
            target=self.cell(*self.matched_goal)
        elif self.t%12==1 or not self.path:
            unknown=cv2.dilate((self.grid==0).astype(np.uint8),np.ones((3,3),np.uint8))>0
            frontier=safe&unknown
            n,labels,stats,centers=cv2.connectedComponentsWithStats(frontier.astype(np.uint8),8)
            candidates=[]
            for label in range(1,n):
                if stats[label,cv2.CC_STAT_AREA]<3:continue
                coords=np.argwhere(labels==label);d=np.linalg.norm(coords-np.array(start),axis=1)
                k=int(np.argmin(d));r,c=coords[k]
                if d[k]<5:continue
                novelty=float(self.visits[max(0,r-4):r+5,max(0,c-4):c+5].sum())
                utility=math.sqrt(len(coords))/(1.+d[k]*.03)/(1.+novelty*.1)
                candidates.append((utility,(int(r),int(c))))
            for _,candidate in sorted(candidates,reverse=True)[:12]:
                path=self.astar(safe,start,candidate)
                if len(path)>3:self.path=path;break
        if target is not None:self.path=self.astar(safe,start,target)
        while self.path and math.hypot(self.path[0][0]-start[0],self.path[0][1]-start[1])<2:self.path.pop(0)
        if not self.path:
            action=2 if self.t%16<12 else 1
        else:
            rr,cc=self.path[min(3,len(self.path)-1)];wx,wz=self.world(rr,cc)
            desired=math.atan2(-(wx-x),-(wz-z));delta=(desired-yaw+math.pi)%(2*math.pi)-math.pi
            action=1 if abs(delta)<math.radians(18) else (2 if delta>0 else 3)
        self.pending=action;return action,info

def main():
    torch.set_num_threads(1);torch.manual_seed(20261015);cv2.setRNGSeed(20261015);cv2.setNumThreads(1)
    model=Modular()
    meta={'ready':True,'model':'SuperPoint-LightGlue-frontier-adapted','weight_coverage_audit':model.weight_audit,
          'input_track':'RGB-D + perfect relative odometry','not_official_paper_reproduction':True,
          'privileged_target_or_map_inputs':False,'perfect_odometry_from_simulator':True,'target_mask_input':False,'target_depth_input':False,
          'weights':{n:hashlib.file_digest((ROOT/'weights'/n).open('rb'),'sha256').hexdigest() for n in ['superpoint_v1.pth','superpoint_lightglue.pth']}}
    print(json.dumps(meta),flush=True)
    for line in sys.stdin:
        try:
            req=json.loads(line)
            if req['op']=='close':break
            if req['op']=='reset':
                model.reset(np.load(req['goal']),req['task']);print(json.dumps({'ok':'reset'}),flush=True);continue
            start=time.perf_counter();action,diag=model.act(np.load(req['rgb']),np.load(req['depth']),req['odometry'])
            print(json.dumps({'action':action,'latency_ms':(time.perf_counter()-start)*1000,'diagnostics':diag}),flush=True)
        except Exception as e:
            import traceback;traceback.print_exc(file=sys.stderr);print(json.dumps({'error':repr(e)}),flush=True)

if __name__=='__main__':main()
