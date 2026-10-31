"""Additive gesture UI/control only; normal tracking is in the unchanged baseline."""
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

@dataclass
class Hand:
    center: tuple
    size: float
    side: str
    side_score: float
    gesture: str
    score: float
    points: list


class HandCamera:
    """MediaPipe LIVE_STREAM recognition: inference never blocks the control loop."""
    def __init__(self, model):
        import mediapipe as mp
        if not Path(model).is_file():
            raise RuntimeError('Gesture model missing. Run setup_gestures.py in this folder first.')
        self.mp = mp
        self.lock = threading.Lock()
        self.stamp = -1.
        self.hands = []
        self.last_submit = -1
        self.last_time = -100.0
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=2, min_hand_detection_confidence=.7,
            min_hand_presence_confidence=.7, min_tracking_confidence=.7,
            result_callback=self._result)
        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

    def _result(self, result, image, stamp):
        hands = []
        for index, marks in enumerate(result.hand_landmarks):
            points = [(p.x, p.y) for p in marks]
            palm = np.array([points[k] for k in (0, 5, 9, 13, 17)])
            center = tuple(np.mean(palm, axis=0))
            size = float(np.linalg.norm(np.array(points[0]) - points[9]))
            side = result.handedness[index][0]
            categories = result.gestures[index]
            gesture = categories[0].category_name if categories else 'None'
            score = categories[0].score if categories else 0.
            hands.append(Hand(center, size, side.category_name, side.score,
                              gesture, score, points))
        with self.lock:
            self.hands = hands
            self.stamp = stamp / 1000.

    def submit(self, frame, now):
        if now-self.last_time < .1:
            return
        self.last_time = now
        stamp = int(now * 1000)
        if stamp <= self.last_submit:
            return
        self.last_submit = stamp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.recognizer.recognize_async(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), stamp)

    def snapshot(self):
        with self.lock:
            return self.stamp, list(self.hands)

    def close(self):
        self.recognizer.close()


class LatchedPolicy:
    """Hand selection controls command acceptance, never tracking permission."""
    def __init__(self):
        self.selected = None
        self.last_stamp = -1.
        self.raw = 'None'
        self.score = 0.
        self.name = 'None'
        self.since = 0.
        self.count = 0
        self.fired = False
        self.fist_until = -1.
        self.abort_blocked = False
        self.release_since = None
        self.reason = 'H then click the controlling hand'

    def sequence_reset(self):
        self.fist_until = -1.
        self.name = 'None'
        self.count = 0
        self.fired = False

    def lose(self, reason):
        self.selected = None
        self.raw = 'None'
        self.score = 0.
        self.sequence_reset()
        self.reason = reason+'; H then click again (tracking latch unchanged)'

    def select(self, hand, stamp, now):
        if now-stamp > .25 or hand.side_score < .7:
            return False
        self.selected = hand
        self.last_stamp = stamp
        self.abort_blocked = False
        self.sequence_reset()
        self.reason = 'Selected: hold fist, then open palm to START'
        return True

    def update(self, stamp, hands, now):
        if self.selected is None:
            return None
        if now-stamp > .25:
            self.lose('Hand result stale')
            return None
        if stamp <= self.last_stamp:
            return None
        self.last_stamp = stamp
        matches = []
        old = self.selected
        for hand in hands:
            distance = float(np.linalg.norm(np.asarray(hand.center)-old.center))
            ratio = hand.size/max(old.size,.001)
            if (hand.side == old.side and hand.side_score >= .7
                and .5 < ratio < 2 and distance < max(.08,old.size*1.4)):
                matches.append((distance,hand))
        matches.sort(key=lambda x:x[0])
        if not matches or (len(matches)>1 and matches[1][0]-matches[0][0]<.04):
            self.lose('Selected-hand continuity uncertain')
            return None
        hand = matches[0][1]
        self.selected = hand
        self.raw,self.score = hand.gesture,hand.score
        threshold = .60 if hand.gesture == 'Open_Palm' else .8
        name = hand.gesture if hand.score >= threshold else 'None'
        if self.abort_blocked:
            if name not in ('Thumb_Down','None'):
                if self.release_since is None:
                    self.release_since = now
                elif now-self.release_since >= .3:
                    self.abort_blocked = False
            else:
                self.release_since = None
        if name != self.name:
            self.name,self.since,self.count,self.fired = name,now,1,False
        else:
            self.count += 1
        if name == 'Thumb_Down' and now-self.since >= .4 and self.count >= 3:
            self.fist_until = -1.
            if not self.fired and not self.abort_blocked:
                self.fired = True
                self.reason = 'ABORT accepted'
                return 'ABORT'
        if name == 'Closed_Fist' and now-self.since >= .65 and self.count >= 3:
            self.fist_until = now+4.
            self.reason = 'Fist confirmed: open palm within 4 seconds'
        if (name == 'Open_Palm' and now-self.since >= .65 and self.count >= 3
            and now < self.fist_until and not self.fired):
            self.fired = True
            self.fist_until = -1.
            self.reason = 'START sequence accepted'
            return 'START'
        return None


HOME = np.array([90.,159.,90.,90.,90.,73.])
POSE_LO = np.array([20.,90.,10.,60.,90.,73.])
POSE_HI = np.array([160.,165.,150.,105.,90.,73.])


def valid_pose(value, lo=POSE_LO, hi=POSE_HI):
    pose = np.asarray(value,float)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError('Pose must contain six finite command angles')
    if np.any(pose < lo) or np.any(pose > hi):
        raise ValueError('Pose outside confirmed command limits')
    return pose


class Routes:
    def __init__(self, path, lo, hi):
        self.path = Path(path)
        self.lo,self.hi = POSE_LO.copy(),POSE_HI.copy()
        self.lo[[0,2]],self.hi[[0,2]] = lo,hi
        self.data = {'routes':[]}
        if self.path.is_file():
            self.data = json.loads(self.path.read_text())
        if not isinstance(self.data,dict) or not isinstance(self.data.get('routes'),list):
            raise ValueError('Invalid abort route file')
        for route in self.data['routes']:
            if not isinstance(route,dict) or np.asarray(route.get('entry',[])).shape!=(6,):
                raise ValueError('Invalid entry in abort route file; existing file left untouched')

    def validate(self, route):
        if route.get('physically_verified') is not True:
            raise ValueError('Abort path has not been physically verified')
        entry = valid_pose(route['entry'],self.lo,self.hi)
        points = [valid_pose(p,self.lo,self.hi) for p in route['waypoints']]
        if not points or len(points)>5000 or np.max(np.abs(points[-1]-HOME))>.02:
            raise ValueError('Abort path must end at the confirmed home pose')
        previous = entry
        for point in points:
            if np.max(np.abs(point-previous))>1.02:
                raise ValueError('Abort waypoints must be at most one degree apart')
            previous = point
        return entry,points

    def match(self, pose):
        options = []
        for route in self.data['routes']:
            try:
                entry,points = self.validate(route)
                if np.max(np.abs(entry-pose)) <= .02:
                    options.append(points)
            except (ValueError,KeyError,TypeError):
                continue
        if not options:
            return None
        # A duplicate entry with differing paths is ambiguous: hold instead.
        if any(len(p)!=len(options[0]) or not np.allclose(p,options[0],atol=.001,rtol=0)
               for p in options[1:]):
            return None
        return [p.copy() for p in options[0]]

    def add_taught(self, entry, points):
        route = {'entry':list(map(float,entry)),
                 'waypoints':[list(map(float,p)) for p in points],
                 'physically_verified':True,
                 'method':'Operator executed this exact one-degree path and pressed V',
                 'recorded_at':time.strftime('%Y-%m-%d %H:%M:%S')}
        self.validate(route)
        self.data['routes'] = [r for r in self.data['routes']
            if not np.allclose(np.asarray(r['entry']),entry,atol=.02,rtol=0)]
        self.data['routes'].append(route)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.data,indent=2))
        temporary.replace(self.path)


def make_link(base, gate):
    class GestureLink(base):
        def __init__(self,*args,**kwargs):
            self._extra_buffer = ''
            self.pose = None
            self.pose_at = -1.
            self.home_id = None
            self.home_sent = 0.
            self.capable = False
            super().__init__(*args,**kwargs)
            original = self.s
            owner = self
            class Tee:
                def read(self,n):
                    data = original.read(n)
                    owner._extra_buffer += data.decode(errors='replace')
                    return data
                def __getattr__(self,name):
                    return getattr(original,name)
            self.s = Tee()
            self.send('I')
            deadline = time.monotonic()+1.2
            while not self.capable and time.monotonic()<deadline:
                self.poll()
                time.sleep(.01)
            if not self.capable:
                self.s.close()
                raise RuntimeError('Upload braccio_gesture_latched.ino: abort/pose capability missing')

        def arm(self):
            if gate.allowed and not gate.busy:
                return super().arm()

        def resume(self):
            if gate.allowed and not gate.busy:
                return super().resume()

        def move(self,q):
            return super().move(q) if gate.allowed and not gate.busy else False

        def wrist(self,tapping):
            return super().wrist(tapping) if gate.allowed and not gate.busy else False

        def trim_shoulder(self,direction):
            # Preserve explicit baseline shoulder jogs while normally paused.
            return super().trim_shoulder(direction) if not gate.aborted and not gate.busy else False

        def stop(self,reason):
            self.home_id = None
            return super().stop(reason)

        def abort(self):
            self.stop('Mission aborted')
            self.pose = None
            self.send('B')

        def query_pose(self):
            self.pose = None
            self.send('Q')

        def home_move(self,pose):
            if self.stop_waiting or self.home_id is not None or not self.initialized:
                return False
            pose = valid_pose(pose,gate.routes.lo,gate.routes.hi)
            if self.pose is None or np.max(np.abs(pose-self.pose))>1.02:
                return False
            self.counter += 1
            self.home_id = self.counter
            self.home_sent = time.monotonic()
            self.send('H '+' '.join(f'{x:.2f}' for x in pose)+f' {self.counter}')
            return True

        def poll(self):
            super().poll()
            now = time.monotonic()
            while '\n' in self._extra_buffer:
                row,self._extra_buffer = self._extra_buffer.split('\n',1)
                p = row.strip().split()
                if 'GESTURE_LATCH_V1' in p:
                    self.capable = True
                elif p and p[0]=='POSE' and len(p)==7:
                    self.pose = valid_pose(list(map(float,p[1:])))
                    self.pose_at = now
                    self.q = self.pose[[0,2]].copy()
                    self.shoulder = float(self.pose[1])
                elif p and p[0]=='HDONE' and len(p)==8 and int(p[1])==self.home_id:
                    self.pose = valid_pose(list(map(float,p[2:])))
                    self.pose_at = now
                    self.q = self.pose[[0,2]].copy()
                    self.shoulder = float(self.pose[1])
                    self.home_id = None
            if self.home_id is not None and now-self.home_sent>3:
                self.fault = 'Abort-path acknowledgement timeout'
                self.stop(self.fault)
    return GestureLink


class GestureGate:
    def __init__(self,args,lo,hi):
        self.args = args
        self.policy = LatchedPolicy()
        self.camera = HandCamera(args.gesture_model)
        self.routes = Routes(args.abort_routes,lo,hi)
        self.allowed = False
        self.aborted = False
        self.phase = 'IDLE'
        self.message = 'Select objects; H then click hand; fist then open palm, or F'
        self.last_command = 'NONE'
        self.want_start_until = -1.
        self.selecting = False
        self.invalidate_j = False
        self.last_pose_printed = -1.
        self.selected_joint = 0
        self.teach_points = []
        self.teach_pending = False
        self.return_points = []
        self.return_index = 0
        self.phase_at = 0.
        self.shape = (480,640)
        self.hands = []
        self.stamp = -1.
        cv2.namedWindow('Gesture commands')
        cv2.setMouseCallback('Gesture commands',self.mouse)

    @property
    def busy(self):
        return self.phase in ('ABORT_STOP','ABORT_QUERY','RETURN','TEACH_QUERY','TEACH')

    def mouse(self,event,x,y,flags,parameter):
        if event != cv2.EVENT_LBUTTONDOWN or not self.selecting:
            return
        height,width = self.shape
        if y>=height:
            return
        now = time.monotonic()
        choices = []
        for hand in self.hands:
            points = np.asarray(hand.points)*[width,height]
            margin = 20
            if (points[:,0].min()-margin <= x <= points[:,0].max()+margin
                and points[:,1].min()-margin <= y <= points[:,1].max()+margin):
                distance = np.linalg.norm(np.asarray(hand.center)*[width,height]-[x,y])
                choices.append((distance,hand))
        choices.sort(key=lambda a:a[0])
        ambiguous = len(choices)>1 and choices[1][0]-choices[0][0]<20
        if choices and not ambiguous and self.policy.select(choices[0][1],self.stamp,now):
            self.selecting = False
            self.message = self.policy.reason
        else:
            self.message = 'No fresh confident hand at that click; keep hand visible'

    def abort(self,link,now):
        self.allowed = False
        self.aborted = True
        self.want_start_until = -1.
        self.policy.sequence_reset()
        self.policy.abort_blocked = True
        self.invalidate_j = True
        self.last_command = 'ABORT'
        self.return_points = []
        self.teach_points = []
        self.teach_pending = False
        if link:
            link.abort()
        if link and link.initialized:
            self.phase = 'ABORT_STOP'
            self.phase_at = now
            self.message = 'Mission aborted: stopping and cancelling current actions'
        else:
            self.phase = 'ABORT_HOLD'
            self.message = 'ABORT: idle (recognition-only or robot not initialized)'

    def emergency(self,link):
        changed_pose = self.busy or self.aborted
        self.allowed = False
        self.want_start_until = -1.
        self.policy.sequence_reset()
        self.invalidate_j = self.invalidate_j or changed_pose
        self.return_points = []
        self.teach_points = []
        self.teach_pending = False
        if link:
            link.stop('Keyboard emergency hold')
        self.phase = 'ABORT_HOLD' if self.aborted else 'IDLE'
        self.last_command = 'STOP / HOLD'
        self.message = 'Stopped and holding. No home movement. F deliberately restarts.'

    def frame(self,frame,now,link):
        self.shape = frame.shape[:2]
        self.camera.submit(frame,now)
        self.stamp,self.hands = self.camera.snapshot()
        event = self.policy.update(self.stamp,self.hands,now)
        if event == 'ABORT':
            self.abort(link,now)
        elif event == 'START' and not self.busy:
            self.want_start_until = now+3.
            self.message = 'START requested; waiting briefly for valid object detection'
        self.tick_return(link,now)

    def hold_return(self,link,message):
        if link:
            link.stop(message)
        self.phase = 'ABORT_HOLD'
        self.message = message+'; remaining stopped'
        self.return_points = []
        self.teach_points = []
        self.teach_pending = False

    def tick_return(self,link,now):
        if not link:
            return
        if self.busy and link.fault:
            self.hold_return(link,'Return/teaching fault: '+link.fault)
            return
        if self.phase in ('ABORT_STOP','ABORT_QUERY','TEACH_QUERY') and now-self.phase_at>3:
            self.hold_return(link,'Stopped-pose acknowledgement timeout')
            return
        if self.phase=='ABORT_STOP' and not link.stop_waiting:
            link.query_pose()
            self.phase = 'ABORT_QUERY'
            self.phase_at = now
        elif self.phase=='ABORT_QUERY' and link.pose is not None:
            print('ABORT ENTRY [M1,M2,M3,M4,M5,M6]:',link.pose.tolist(),flush=True)
            if not link.shoulder_enabled:
                self.hold_return(link,'Home return unavailable: M2 is disabled')
            elif np.max(np.abs(link.pose-HOME))<=.02:
                self.phase = 'ABORT_HOLD'
                self.message = 'Already at home; aborted and idle'
            elif self.args.no_home_return:
                self.hold_return(link,'Home return disabled by --no-home-return')
            else:
                points = self.routes.match(link.pose)
                if points is None:
                    self.phase = 'ABORT_HOLD'
                    self.message = 'ABORT HOLD: no verified path from this exact pose. R teaches one.'
                else:
                    self.return_points = points
                    self.return_index = 0
                    self.phase = 'RETURN'
                    self.message = 'Returning slowly along the exact physically verified path'
        elif self.phase=='RETURN' and link.home_id is None and not link.stop_waiting:
            if self.return_index >= len(self.return_points):
                self.phase = 'ABORT_HOLD'
                self.message = 'Home reached: aborted and idle. New fist/palm sequence or F required.'
                self.last_command = 'ABORT / HOME COMPLETE'
                self.policy.sequence_reset()
            elif link.home_move(self.return_points[self.return_index]):
                self.return_index += 1
            else:
                self.hold_return(link,'Verified return waypoint rejected')
        elif self.phase=='TEACH_QUERY' and link.pose is not None:
            self.teach_entry = link.pose.copy()
            self.teach_points = []
            self.teach_pending = False
            self.selected_joint = 0
            self.phase = 'TEACH'
            self.message = 'TEACH: 1/2/3/4 select joint, comma/period jog -/+1; V saves at home'
        elif self.phase=='TEACH' and self.teach_pending and link.home_id is None:
            self.teach_points.append(link.pose.copy())
            self.teach_pending = False
            self.message = 'TEACH: inspect clearance. 1/2/3/4 select; comma/period jog; V saves at home'
        if link.pose is not None and link.pose_at>self.last_pose_printed and not self.busy:
            self.last_pose_printed = link.pose_at
            print('CURRENT COMMAND POSE:',link.pose.tolist(),flush=True)

    def key(self,key,now,link,valid,state):
        if key in (ord('q'),27):
            return key
        if key in (ord(' '),ord('e')):
            self.emergency(link)
            return ord(' ')
        if key==ord('b'):
            self.abort(link,now)
            return ord(' ')
        if key==ord('h'):
            self.selecting = True
            self.policy.lose('Selecting controlling hand')
            self.message = 'Click the controlling hand in Gesture commands (not the green object window)'
            return 255
        if self.busy:
            if self.phase=='TEACH':
                if key in (ord('1'),ord('2'),ord('3'),ord('4')):
                    self.selected_joint = key-ord('1')
                elif key in (ord(','),ord('.')) and link.home_id is None and not self.teach_pending:
                    target = link.pose.copy()
                    target[self.selected_joint] += -1 if key==ord(',') else 1
                    try:
                        if link.home_move(target):
                            self.teach_pending = True
                    except ValueError:
                        self.message = 'Teach jog at limit; no command sent'
                elif key==ord('v') and link.home_id is None and not self.teach_pending:
                    if np.max(np.abs(link.pose-HOME))>.02:
                        self.message = 'Not at home yet. Required: 90,159,90,90,90,73'
                    elif self.teach_points:
                        try:
                            self.routes.add_taught(self.teach_entry,self.teach_points)
                            self.phase = 'ABORT_HOLD'
                            self.message = 'Verified path saved for its EXACT entry pose only; idle'
                        except (ValueError,OSError) as error:
                            self.message = 'Route not saved: '+str(error)
            return 255
        if key==ord('p'):
            if link and link.initialized and not self.allowed and not link.stop_waiting:
                link.query_pose()
            return 255
        if key==ord('r'):
            if self.aborted and link and link.initialized and link.shoulder_enabled and not link.stop_waiting:
                link.query_pose()
                self.phase = 'TEACH_QUERY'
                self.phase_at = now
                self.message = 'Teaching a NEW path: operator must observe every slow jog for clearance'
            else:
                self.message = 'R teaching requires an initialized robot stopped by B/abort'
            return 255
        gesture_start = self.want_start_until>=now
        if key==ord('f') or gesture_start or (key==ord('c') and not self.aborted):
            ready = (valid and (not link or (not link.stop_waiting and link.pending is None
                    and link.shoulder_id is None))
                    and state not in ('TAPPING','RECOVER','CANCEL_WAIT','ARMING','LEARN','SHOULDER'))
            if ready:
                self.allowed = True
                self.aborted = False
                self.phase = 'IDLE'
                self.want_start_until = -1.
                if link:
                    link.send('U')
                self.last_command = 'START: gesture' if gesture_start else 'START: keyboard'
                self.message = 'Following enabled; hand visibility no longer controls movement'
                return key if key==ord('c') else ord('f')
            return 255
        if key==ord('c') and self.aborted:
            self.message = 'Aborted: F or a new fist/open-palm sequence is required'
            return 255
        if self.aborted and key in (ord('['),ord(']')):
            self.message = 'Aborted: use deliberate route teaching, or F to unlock normal controls'
            return 255
        return key

    def draw(self,frame,state):
        view = frame.copy()
        height,width = view.shape[:2]
        hand = self.policy.selected
        if hand:
            p = np.asarray(hand.points)*[width,height]
            x0,y0 = np.maximum(p.min(axis=0)-10,0).astype(int)
            x1,y1 = np.minimum(p.max(axis=0)+10,[width-1,height-1]).astype(int)
            cv2.rectangle(view,(x0,y0),(x1,y1),(0,255,0),2)
            cv2.putText(view,'SELECTED '+hand.side,(x0,max(18,y0-8)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
        mode = 'FOLLOWING ENABLED (latched)' if self.allowed else 'STOPPED / IDLE'
        rows = [mode+' | core '+state+' | '+self.phase,
                f'Gesture: {self.policy.raw} {self.policy.score:.0%}',
                'Last accepted: '+self.last_command,self.message,self.policy.reason,
                'H+click hand | fist then palm START | thumbs down/B ABORT | Space/E HOLD',
                'Hand loss does NOT stop following. Selection is continuity, not identity.']
        if self.phase=='TEACH':
            rows.append(f'TEACH M{self.selected_joint+1}: 1/2/3/4 select, comma/period -/+1, V confirm clear path')
        panel = np.zeros((25*len(rows)+12,width,3),np.uint8)
        for i,row in enumerate(rows):
            cv2.putText(panel,row,(8,22+25*i),cv2.FONT_HERSHEY_SIMPLEX,.41,(230,240,240),1)
        cv2.imshow('Gesture commands',np.vstack([view,panel]))

    def close(self):
        self.camera.close()


