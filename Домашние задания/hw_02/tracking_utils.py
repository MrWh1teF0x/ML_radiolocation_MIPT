"""Shared utilities for the teaching benchmark. No GT identities are passed to trackers."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
from scipy.optimize import linear_sum_assignment

@dataclass
class Frame:
    number: int
    image_path: Path
    gt: np.ndarray  # [n, 6]: x1,y1,x2,y2,track_id,class_id


def sequence_key(stem: str) -> tuple[str, int]:
    match = re.fullmatch(r"(.+)_(\d+)", stem)
    if not match:
        raise ValueError(f"Expected <sequence>_<frame_number>, got {stem!r}")
    return match.group(1), int(match.group(2))


def read_extended_yolo(path: Path, width: int, height: int, require_id: bool = True) -> np.ndarray:
    """Returns pixel xyxy + track_id + class_id. 6th column is track_id, NOT confidence."""
    lines = path.read_text(encoding='utf-8').splitlines() if path.exists() else []
    out = []
    for idx, line in enumerate(lines, 1):
        if not line.strip():
            continue
        vals = [float(x) for x in line.split()]
        if len(vals) != (6 if require_id else 5):
            raise ValueError(f"{path}:{idx}: expected {'6' if require_id else '5'} values, got {len(vals)}")
        cls, xc, yc, w, h = vals[:5]
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"{path}:{idx}: invalid normalized coordinates")
        x1, y1 = (xc - w / 2) * width, (yc - h / 2) * height
        x2, y2 = (xc + w / 2) * width, (yc + h / 2) * height
        tid = int(vals[5]) if require_id else -1
        out.append([x1, y1, x2, y2, tid, int(cls)])
    return np.asarray(out, dtype=float).reshape(-1, 6)


def discover_sequences(split_dir: str | Path) -> dict[str, list[Frame]]:
    """Directory structure: split/images/*.jpg and split/labels/*.txt."""
    from PIL import Image
    root = Path(split_dir)
    result = {}
    for image_path in sorted((root / 'images').glob('*')):
        if image_path.suffix.lower() not in {'.jpg','.jpeg','.png'}:
            continue
        seq, number = sequence_key(image_path.stem)
        with Image.open(image_path) as im:
            width, height = im.size
        gt = read_extended_yolo(root / 'labels' / (image_path.stem + '.txt'), width, height)
        result.setdefault(seq, []).append(Frame(number, image_path, gt))
    if not result:
        raise FileNotFoundError(f"No images in {root/'images'}")
    for frames in result.values():
        frames.sort(key=lambda f: f.number)
        ids = [f.number for f in frames]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate frame numbers')
    return result


def strip_ids(gt: np.ndarray) -> np.ndarray:
    """Ideal measurements: array [n, 5] = xyxy + confidence. No track IDs."""
    return np.column_stack((gt[:, :4], np.ones(len(gt)))).reshape(-1, 5)


def noisy_measurements(gt: np.ndarray, image_size: tuple[int,int], rng: np.random.Generator,
                       p_miss: float = .12, jitter_px: float = 2.,
                       fp_rate: float = .12) -> np.ndarray:
    """Synthetic detections from GT, no GT ids; Poisson false positives."""
    width,height = image_size
    boxes = strip_ids(gt)
    result = []
    for box in boxes:
        if rng.random() < p_miss:
            continue
        x1,y1,x2,y2,conf = box
        cx,cy = (x1+x2)/2,(y1+y2)/2
        w,h = max(1,x2-x1),max(1,y2-y1)
        cx += rng.normal(0,jitter_px);cy += rng.normal(0,jitter_px)
        w *= np.clip(1+rng.normal(0,.08),.5,1.5)
        h *= np.clip(1+rng.normal(0,.08),.5,1.5)
        result.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2,float(rng.uniform(.65,1))])
    for _ in range(rng.poisson(fp_rate)):
        w,h = rng.uniform(5,22),rng.uniform(5,22)
        cx,cy = rng.uniform(0,width),rng.uniform(0,height)
        result.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2,float(rng.uniform(.05,.7))])
    return np.array(result, dtype=float).reshape(-1,5)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a=np.asarray(a,dtype=float).reshape(-1,4)
    b=np.asarray(b,dtype=float).reshape(-1,4)
    if not len(a) or not len(b):return np.zeros((len(a),len(b)))
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:4],b[None,:,2:4])
    inter=np.maximum(hi-lo,0).prod(axis=-1)
    area_a=np.maximum(a[:,2:]-a[:,:2],0).prod(axis=1)
    area_b=np.maximum(b[:,2:]-b[:,:2],0).prod(axis=1)
    return inter/np.maximum(area_a[:,None]+area_b[None,:]-inter,1e-12)


def match_cost(cost: np.ndarray, allowed: np.ndarray) -> tuple[list[tuple[int,int]],list[int],list[int]]:
    """Hungarian matching plus explicit gating; forbidden costs cannot be selected."""
    m,n=cost.shape
    if not m or not n:return [],list(range(m)),list(range(n))
    safe=np.where(allowed,cost,1e9)
    rr,cc=linear_sum_assignment(safe)
    pairs=[(int(i),int(j)) for i,j in zip(rr,cc) if allowed[i,j]]
    used_r={i for i,_ in pairs};used_c={j for _,j in pairs}
    return pairs,[i for i in range(m) if i not in used_r],[j for j in range(n) if j not in used_c]


class KalmanBox:
    """Constant-velocity filter over [cx,cy,vx,vy,w,h] in image pixels; dt=frame gap."""
    def __init__(self, bbox: np.ndarray, pos_var=25., vel_var=16., meas_var=9., accel_var=2.):
        x1,y1,x2,y2=np.asarray(bbox,dtype=float)[:4]
        self.x=np.array([(x1+x2)/2,(y1+y2)/2,0,0,max(x2-x1,1),max(y2-y1,1)],dtype=float)
        self.P=np.diag([pos_var,pos_var,vel_var,vel_var,25,25]).astype(float)
        self.R=np.eye(4)*meas_var
        self.H=np.zeros((4,6));self.H[0,0]=self.H[1,1]=self.H[2,4]=self.H[3,5]=1
        self.accel_var=accel_var
    def predict(self, dt=1):
        F=np.eye(6);F[0,2]=F[1,3]=dt
        # White-acceleration process noise on center/velocity; size random walk.
        G=np.array([[.5*dt**2,0],[0,.5*dt**2],[dt,0],[0,dt],[0,0],[0,0]])
        Q=self.accel_var*(G@G.T)+np.diag([.01,.01,.01,.01,.2*dt,.2*dt])
        self.x=F@self.x;self.P=F@self.P@F.T+Q
        return self.box()
    def update(self,bbox):
        x1,y1,x2,y2=np.asarray(bbox,dtype=float)[:4]
        z=np.array([(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1])
        S=self.H@self.P@self.H.T+self.R
        K=np.linalg.solve(S,(self.P@self.H.T).T).T
        self.x+=K@(z-self.H@self.x)
        # Joseph covariance update is robust to numerical round-off.
        A=np.eye(6)-K@self.H
        self.P=A@self.P@A.T+K@self.R@K.T
        self.x[4:]=np.maximum(self.x[4:],1.)
    def box(self):
        cx,cy,_,_,w,h=self.x
        return np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2])


class SortLikeTracker:
    """Educational SORT-like bbox tracker; IDs start at 1 independently per sequence."""
    def __init__(self,association='iou',iou_threshold=.1,distance_threshold=35.,
                 max_age=3,min_hits=1,meas_var=9.,accel_var=2.):
        if association not in {'iou','center'}:raise ValueError('association must be iou or center')
        self.association=association;self.iou_threshold=iou_threshold
        self.distance_threshold=distance_threshold;self.max_age=max_age;self.min_hits=min_hits
        self.meas_var=meas_var;self.accel_var=accel_var
        self.tracks=[];self.next_id=1;self.previous_frame=None
    def step(self, detections:np.ndarray, frame_number:int) -> np.ndarray:
        """Input [n,5] xyxy+score; output [k,5] xyxy+predicted_track_id (observed tracks only)."""
        det=np.asarray(detections,dtype=float).reshape(-1,5)
        if self.previous_frame is not None and frame_number <= self.previous_frame:
            raise ValueError('Frame numbers must increase')
        dt=1 if self.previous_frame is None else frame_number-self.previous_frame
        self.previous_frame=frame_number
        for t in self.tracks:
            t['kf'].predict(dt);t['age']+=dt;t['seen']=False
        tb=np.asarray([t['kf'].box() for t in self.tracks],dtype=float).reshape(-1,4)
        db=det[:,:4]
        if self.association=='iou':
            sims=bbox_iou(tb,db);cost=1-sims;allowed=sims>=self.iou_threshold
        else:
            tc=(tb[:,:2]+tb[:,2:])/2;dc=(db[:,:2]+db[:,2:])/2
            cost=np.linalg.norm(tc[:,None,:]-dc[None,:,:],axis=-1)
            allowed=cost<=self.distance_threshold
        matches,_,unmatched_det=match_cost(cost,allowed)
        for i,j in matches:
            t=self.tracks[i];t['kf'].update(db[j]);t['age']=0;t['hits']+=1;t['seen']=True
        for j in unmatched_det:
            t=dict(kf=KalmanBox(db[j],meas_var=self.meas_var,accel_var=self.accel_var),
                   id=self.next_id,age=0,hits=1,seen=True)
            self.next_id+=1;self.tracks.append(t)
        self.tracks=[t for t in self.tracks if t['age']<=self.max_age]
        visible=[[*t['kf'].box(),t['id']] for t in self.tracks if t['seen'] and t['hits']>=self.min_hits]
        return np.asarray(visible,dtype=float).reshape(-1,5)


def run_tracker(frames:list[Frame], tracker:SortLikeTracker,
                mode='ideal',seed=42, image_size=(640,512)) -> list[np.ndarray]:
    rng=np.random.default_rng(seed)
    outputs=[]
    for frame in frames:
        if mode=='ideal':detections=strip_ids(frame.gt)
        elif mode=='noisy':detections=noisy_measurements(frame.gt,image_size,rng)
        else:raise ValueError('Use ideal or noisy; detector requires external detections')
        outputs.append(tracker.step(detections,frame.number))
    return outputs


def save_mot(path:str|Path,frames:list[Frame],predictions:list[np.ndarray]):
    """MOTChallenge-like 10-col text: frame,id,x,y,w,h,score,-1,-1,-1; 1-based frame indices."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8') as f:
        for frame,pred in zip(frames,predictions):
            for x1,y1,x2,y2,tid in pred:
                f.write(f'{frame.number},{int(tid)},{x1:.4f},{y1:.4f},{x2-x1:.4f},{y2-y1:.4f},1,-1,-1,-1\n')


def _frame_pairs(gt:np.ndarray,pred:np.ndarray,threshold=.5):
    similarity=bbox_iou(gt[:,:4],pred[:,:4]);matches,_,_=match_cost(1-similarity,similarity>=threshold)
    return matches


def teaching_mot_metrics(frames:list[Frame],predictions:list[np.ndarray],threshold=.5) -> dict:
    """Transparent introductory metrics. For publication use standard TrackEval implementation."""
    if len(frames)!=len(predictions):raise ValueError('Frame count mismatch')
    gt_total=sum(len(f.gt) for f in frames);pred_total=sum(len(p) for p in predictions)
    tp=0;idsw=0;last_match={};pair_counts={};all_gt=set();all_pred=set()
    for f,p in zip(frames,predictions):
        gt=f.gt; p=np.asarray(p).reshape(-1,5)
        all_gt.update(map(int,gt[:,4]));all_pred.update(map(int,p[:,4]))
        for i,j in _frame_pairs(gt,p,threshold):
            tp+=1;gid=int(gt[i,4]);pid=int(p[j,4])
            if gid in last_match and last_match[gid]!=pid:idsw+=1
            last_match[gid]=pid
            pair_counts[(gid,pid)]=pair_counts.get((gid,pid),0)+1
    gids=sorted(all_gt);pids=sorted(all_pred)
    mat=np.array([[pair_counts.get((g,p),0) for p in pids] for g in gids],dtype=float)
    idtp=0
    if mat.size:
        ri,ci=linear_sum_assignment(-mat);idtp=int(mat[ri,ci].sum())
    fp=pred_total-tp;fn=gt_total-tp
    return dict(TP=tp,FP=fp,FN=fn,IDSW=idsw,
                MOTA=1-(fn+fp+idsw)/gt_total if gt_total else float('nan'),
                IDF1=2*idtp/(gt_total+pred_total) if gt_total+pred_total else float('nan'),
                precision=tp/pred_total if pred_total else 0.,recall=tp/gt_total if gt_total else 0.)


def hota_metrics(frames:list[Frame],predictions:list[np.ndarray],alphas=None)->dict:
    """HOTA/DetA/AssA (bbox IoU), following global alignment + per-alpha Hungarian protocol.
    Teaching implementation: verify against TrackEval for formal benchmarking.
    """
    if len(frames)!=len(predictions):raise ValueError('Frame count mismatch')
    if alphas is None:alphas=np.arange(.05,.96,.05)
    gids=sorted({int(x) for f in frames for x in f.gt[:,4]})
    pids=sorted({int(x) for p in predictions for x in np.asarray(p).reshape(-1,5)[:,4]})
    gi={v:i for i,v in enumerate(gids)};pi={v:i for i,v in enumerate(pids)}
    ng=len(gids);npred=len(pids)
    gt_count=np.zeros(ng);pred_count=np.zeros(npred);potential=np.zeros((ng,npred))
    packed=[]
    for f,p in zip(frames,predictions):
        gt=f.gt;p=np.asarray(p,dtype=float).reshape(-1,5)
        gidx=np.array([gi[int(v)] for v in gt[:,4]],dtype=int)
        pidx=np.array([pi[int(v)] for v in p[:,4]],dtype=int)
        np.add.at(gt_count,gidx,1);np.add.at(pred_count,pidx,1)
        sim=bbox_iou(gt[:,:4],p[:,:4]);packed.append((gidx,pidx,sim))
        if sim.size:
            denom=sim.sum(axis=1,keepdims=True)+sim.sum(axis=0,keepdims=True)-sim
            weighted=np.divide(sim,denom,out=np.zeros_like(sim),where=denom>0)
            np.add.at(potential,(gidx[:,None],pidx[None,:]),weighted)
    denom=gt_count[:,None]+pred_count[None,:]-potential
    global_alignment=np.divide(potential,denom,out=np.zeros_like(potential),where=denom>0)
    per_alpha=[]
    gt_total=int(gt_count.sum());pr_total=int(pred_count.sum())
    for alpha in alphas:
        tp=0;matched=np.zeros((ng,npred),dtype=int)
        for gidx,pidx,sim in packed:
            if not sim.size:continue
            score=global_alignment[np.ix_(gidx,pidx)]*sim
            pairs,_,_=match_cost(-score,sim>=alpha-1e-10)
            tp+=len(pairs)
            for i,j in pairs:matched[gidx[i],pidx[j]]+=1
        fn=gt_total-tp;fp=pr_total-tp
        deta=tp/(tp+fn+fp) if tp+fn+fp else 1.
        ass_denom=gt_count[:,None]+pred_count[None,:]-matched
        pair_ass=np.divide(matched,ass_denom,out=np.zeros_like(ass_denom,dtype=float),where=ass_denom>0)
        assa=(matched*pair_ass).sum()/tp if tp else 0.
        per_alpha.append((float(np.sqrt(deta*assa)),float(deta),float(assa)))
    result=np.mean(per_alpha,axis=0)
    return dict(HOTA=float(result[0]),DetA=float(result[1]),AssA=float(result[2]),
                alpha_05=dict(zip(['HOTA','DetA','AssA'],per_alpha[0])))


def evaluate(frames:list[Frame], predictions:list[np.ndarray],iou_threshold=.5)->dict:
    return {**teaching_mot_metrics(frames,predictions,iou_threshold),**hota_metrics(frames,predictions)}
