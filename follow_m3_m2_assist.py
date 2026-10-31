"""
Braccio five-tap follower with automatic full-frame reacquisition.

Use the matching braccio_shoulder_fallback Arduino sketch.
M2 holds the required --shoulder-start angle. [ and ] manually trim while paused.
--shoulder-assist enables one-degree M2 help when M3 is at its blocking limit.
M2 stays outside the original M1/M3 solver; each assistance step is checked for benefit. Range: 90-165 degrees.
--no-shoulder leaves D5 unattached; mechanically support M2 in that mode.

Click the gripper marker first, then the target.
Automatic startup after a three-second countdown.

SPACE: pause
F: resume
C: relearn directions
T: toggle tapping
X: reselect objects
O: save current gap
Z: clear saved gap
Q: quit
"""
import argparse
import itertools
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
WINDOW = 'Braccio: two green objects'
OUTER_LO = np.array([20.0, 10.0])
OUTER_HI = np.array([160.0, 150.0])
SETTINGS_FILE = Path(__file__).with_name('follower_fast_settings.json')

def bounded_step(J, error, q, lo, hi):
    if np.linalg.norm(error) <= 1:
        return np.zeros(2)
    A = np.asarray(J, float)
    desired = 0.5 * np.asarray(error, float)
    step = np.linalg.solve(A.T @ A + np.eye(2) * 4, A.T @ desired)
    step *= min(1.0, 4.0 / max(np.max(np.abs(step)), 1e-08))
    for j in range(2):
        blocked = q[j] <= lo[j] + 0.1 and step[j] < 0 or (q[j] >= hi[j] - 0.1 and step[j] > 0)
        if blocked:
            step[j] = 0
            k = 1 - j
            step[k] = np.clip(np.dot(A[:, k], desired) / (np.dot(A[:, k], A[:, k]) + 4), -4, 4)
    return np.clip(q + step, lo, hi) - q

def assess_progress(old_marker, marker, old_target, target, offset, count):
    """Judge servo response against the OLD target, excluding target motion."""
    target_shift = float(np.linalg.norm(target - old_target))
    before = np.linalg.norm(old_target + offset - old_marker)
    after = np.linalg.norm(old_target + offset - marker)
    change = float(after - before)
    if target_shift > 8.0:
        return (0, target_shift, change)
    return (count + 1 if change > 4.0 else max(0, count - 1), target_shift, change)

class Vision:

    def __init__(self):
        self.models = [None, None]
        self.prev = [None, None]
        self.area = [None, None]
        self.candidates = []
        self.problem = 'Click gripper marker first, then target'
        self.good = 0
        self.recovery_pair = None
        self.recovery_count = 0
        self.recovering = False
        self.kernel = np.ones((3, 3), np.uint8)
        cv2.namedWindow('Green settings')
        settings = {'H min': 30, 'H max': 95, 'S min': 50, 'V min': 40, 'Min area': 80, 'Hue tolerance': 14, 'Match tolerance': 16}
        try:
            old = json.loads((Path(__file__).parent / 'green_hsv.json').read_text())
            for key in ('H min', 'H max', 'S min', 'V min'):
                if key in old:
                    settings[key] = int(old[key])
        except (OSError, ValueError):
            pass
        for key, value in settings.items():
            maximum = 179 if key.startswith('H ') else 2000 if key == 'Min area' else 255
            cv2.createTrackbar(key, 'Green settings', value, maximum, lambda _: None)

    def reset(self):
        self.models = [None, None]
        self.prev = [None, None]
        self.area = [None, None]
        self.good = 0
        self.recovery_pair = None
        self.recovery_count = 0
        self.recovering = False

    def click(self, x, y):
        role = 0 if self.models[0] is None else 1
        if self.models[1] is not None:
            return
        choices = [candidate for candidate in self.candidates if cv2.pointPolygonTest(candidate['contour'], (float(x), float(y)), False) >= 0]
        if not choices:
            self.problem = 'Click INSIDE a green object; adjust Green settings'
            return
        candidate = max(choices, key=lambda item: item['area'])
        if role == 1:
            if np.linalg.norm(candidate['p'] - self.prev[0]) < 15:
                self.problem = 'Choose a DIFFERENT green object for the target'
                return
        self.models[role] = candidate['hsv'].copy()
        self.prev[role] = candidate['p'].copy()
        self.area[role] = candidate['area']
        self.problem = 'Now click target' if role == 0 else 'Both selected. Automatic start shortly'
        self.good = 0

    def frame(self, image):

        def get(name):
            return cv2.getTrackbarPos(name, 'Green settings')
        hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([get('H min'), get('S min'), get('V min')]), np.array([get('H max'), 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max(15, get('Min area')):
                continue
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
            contour_mask = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            pixels = hsv[(contour_mask > 0) & (mask > 0)]
            self.candidates.append({'p': np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']]), 'area': area, 'hsv': np.median(pixels, axis=0), 'contour': contour})
        if any((model is None for model in self.models)):
            self.good = 0
            return (None, mask)
        pair = self.match_candidates(max(2, get('Hue tolerance')))
        return (pair, mask)

    def match_candidates(self, hue_tolerance):

        def assign(global_search):
            costs = np.full((2, len(self.candidates)), np.inf)
            for role in range(2):
                for index, candidate in enumerate(self.candidates):
                    dh = abs(candidate['hsv'][0] - self.models[role][0])
                    dh = min(dh, 180 - dh)
                    ds = abs(candidate['hsv'][1] - self.models[role][1])
                    dv = abs(candidate['hsv'][2] - self.models[role][2])
                    distance = np.linalg.norm(candidate['p'] - self.prev[role])
                    ratio = candidate['area'] / self.area[role]
                    if not (dh <= hue_tolerance and ds < 110 and (dv < 110) and (0.45 < ratio < 2.2)):
                        continue
                    if global_search:
                        score = dh / 6 + ds / 80 + dv / 100 + 2 * abs(np.log(ratio))
                        if score < 4:
                            costs[role, index] = score
                    elif distance < 75:
                        costs[role, index] = distance / 40 + dh / 12 + ds / 100 + dv / 120 + abs(np.log(ratio))
            assignments = sorted(((costs[0, first] + costs[1, second], first, second) for first, second in itertools.permutations(range(len(self.candidates)), 2)))
            margin = 0.7 if global_search else 0.45
            if not assignments or not np.isfinite(assignments[0][0]) or (len(assignments) > 1 and assignments[1][0] - assignments[0][0] < margin):
                return None
            _, first, second = assignments[0]
            pair = [self.candidates[first], self.candidates[second]]
            if np.linalg.norm(pair[0]['p'] - pair[1]['p']) < 12:
                return None
            return pair
        pair = None if self.recovering else assign(False)
        if pair is None:
            self.recovering = True
            self.good = 0
            pair = assign(True)
            if pair is None:
                self.recovery_pair = None
                self.recovery_count = 0
                self.problem = 'SEARCHING FULL IMAGE: waiting for two distinct objects'
                return None
            consistent = self.recovery_pair is not None and all((np.linalg.norm(pair[k]['p'] - self.recovery_pair[k]['p']) < 45 and 0.65 < pair[k]['area'] / self.recovery_pair[k]['area'] < 1.55 for k in range(2)))
            self.recovery_count = self.recovery_count + 1 if consistent else 1
            self.recovery_pair = pair
            if self.recovery_count < 3:
                self.problem = f'TARGET FOUND: confirming identity {self.recovery_count}/3'
                return None
            self.recovering = False
            self.recovery_pair = None
            self.recovery_count = 0
            self.good = 6
            self.problem = 'TARGET REACQUIRED: automatic following'
        else:
            self.good += 1
        for role, candidate in enumerate(pair):
            self.prev[role] = candidate['p'].copy()
            self.area[role] = 0.98 * self.area[role] + 0.02 * candidate['area']
        return pair

class Link:

    def __init__(self, port, q, shoulder_enabled=True, shoulder_start=90):
        import serial
        self.s = serial.Serial(port, 115200, timeout=0, write_timeout=0.15)
        self.shoulder_enabled = shoulder_enabled
        self.shoulder = float(shoulder_start)
        self.shoulder_id = None
        self.shoulder_started = 0.0
        self.buffer = ''
        self.pending = None
        self.counter = 0
        self.q = np.asarray(q, float)
        self.done_at = 0.0
        self.sent_at = 0.0
        self.last_beat = 0.0
        self.fault = ''
        self.armed = False
        self.initialized = False
        self.wrist_id = None
        self.wrist_started = 0.0
        self.stop_waiting = False
        self.stop_sent = 0.0
        deadline = time.monotonic() + 6
        found = False
        while time.monotonic() < deadline:
            self.send('I')
            time.sleep(0.15)
            data = self.s.read(self.s.in_waiting).decode(errors='replace')
            if 'BRACCIO_SHOULDER_FALLBACK_V1' in data:
                found = True
                break
        if not found:
            self.s.close()
            raise RuntimeError('Upload the matching braccio_shoulder_fallback Arduino sketch first: handshake missing.')

    def send(self, message):
        self.s.write((message + '\n').encode('ascii'))

    def arm(self):
        self.send('A %d %d %d %d' % (*self.q.astype(int), int(self.shoulder_enabled), int(self.shoulder)))

    def trim_shoulder(self, direction):
        # Explicit, unqueued manual command; never part of automatic tracking.
        if (not self.shoulder_enabled or not self.initialized or self.armed
            or self.stop_waiting or self.shoulder_id is not None
            or self.pending is not None or self.wrist_id is not None or self.fault):
            return False
        target = float(np.clip(self.shoulder + direction, 90, 165))
        if abs(target - self.shoulder) < .01:
            return False
        self.counter += 1
        self.shoulder_id = self.counter
        self.shoulder_started = time.monotonic()
        self.send(f'J {target:.2f} {self.counter}')
        return True

    def stop(self, reason):
        if self.stop_waiting:
            return
        print(time.strftime('[%H:%M:%S]'), 'STOP:', reason, flush=True)
        self.send('S')
        self.shoulder_id = None
        self.stop_waiting = True
        self.stop_sent = time.monotonic()
        self.wrist_id = None
        self.pending = None
        self.armed = False

    def resume(self):
        if not self.stop_waiting and self.shoulder_id is None:
            self.send('R')
            self.armed = True

    def move(self, q):
        if self.pending is not None or self.wrist_id is not None or self.shoulder_id is not None or (not self.armed):
            return False
        q = np.rint(q).astype(int)
        if np.any(q < OUTER_LO) or np.any(q > OUTER_HI):
            raise RuntimeError('Joint bounds violated')
        self.counter += 1
        self.pending = self.counter
        self.sent_at = time.monotonic()
        self.send(f'M {q[0]} 90 {q[1]} 90 90 73 50 {self.counter}')
        return True

    def wrist(self, tapping):
        if self.pending is not None or self.wrist_id is not None or self.shoulder_id is not None or (not self.armed):
            return False
        self.counter += 1
        self.wrist_id = self.counter
        self.wrist_started = time.monotonic()
        operation = 'T' if tapping else 'N'
        self.send(f'{operation} {self.counter}')
        return True

    def poll(self):
        now = time.monotonic()
        if now - self.last_beat > 0.15:
            self.send('K')
            self.last_beat = now
        self.buffer += self.s.read(self.s.in_waiting).decode(errors='replace')
        while '\n' in self.buffer:
            row, self.buffer = self.buffer.split('\n', 1)
            parts = row.strip().split()
            if not parts:
                continue
            print('Arduino:', row)
            if parts[0] == 'SPOS' and len(parts) == 2:
                self.shoulder = float(parts[1])
            elif parts[0] == 'SDONE' and len(parts) == 3:
                if int(parts[1]) == self.shoulder_id:
                    self.shoulder = float(parts[2])
                    self.shoulder_id = None
            elif parts[0] == 'WDONE' and len(parts) == 2:
                if int(parts[1]) == self.wrist_id:
                    self.wrist_id = None
                    self.done_at = now
            elif parts[0] == 'DONE' and len(parts) == 4:
                if int(parts[1]) == self.pending:
                    self.q = np.array([float(parts[2]), float(parts[3])])
                    self.pending = None
                    self.done_at = now
            elif parts[0] in ('ARMED', 'STOPPED', 'POS', 'WATCHDOG') and len(parts) == 3:
                self.q = np.array([float(parts[1]), float(parts[2])])
                self.pending = None
                self.done_at = now
                if parts[0] == 'STOPPED':
                    self.stop_waiting = False
                if parts[0] == 'ARMED':
                    self.armed = True
                if parts[0] in ('STOPPED', 'WATCHDOG'):
                    self.shoulder_id = None
                if parts[0] == 'WATCHDOG':
                    self.wrist_id = None
                    self.armed = False
                    print('ARDUINO WATCHDOG: communication heartbeat interrupted', flush=True)
                    self.fault = 'Communication paused: C retries, X reselects'
            elif parts[0] in ('BAD', 'IGNORED', 'LIMIT', 'BUFFER'):
                self.fault = row
                self.stop(self.fault)
        if self.shoulder_id is not None and now - self.shoulder_started > 2:
            self.fault = 'Shoulder acknowledgement timeout'
            self.stop(self.fault)
        if self.stop_waiting and now - self.stop_sent > 2:
            self.fault = 'Stop acknowledgement timeout; check connection'
        if self.pending is not None and now - self.sent_at > 3:
            self.fault = 'No DONE acknowledgement'
            self.stop(self.fault)
        if self.wrist_id is not None and now - self.wrist_started > 7:
            self.fault = 'Wrist acknowledgement timeout'
            self.stop(self.fault)

    def close(self):
        try:
            self.stop('Program closing')
        finally:
            self.s.close()

class Learner:

    def __init__(self, link, lo, hi):
        self.link = link
        self.lo = lo
        self.hi = hi
        self.stage = 0
        self.origin = link.q.copy()
        self.start = None
        self.J = np.zeros((2, 2))
        self.error = ''
        self.j = 0
        self.delta = 0

    def tick(self, position, now):
        if self.link.pending is not None or now - self.link.done_at < 0.3:
            return False
        if self.stage == 0:
            self.start = position.copy()
            self.delta = 3 if self.origin[self.j] + 3 <= self.hi[self.j] else -3
            test_angle = self.origin[self.j] + self.delta
            if not self.lo[self.j] <= test_angle <= self.hi[self.j]:
                self.error = 'Not enough space for 3 degree probe'
                return False
            q = self.origin.copy()
            q[self.j] += self.delta
            if self.link.move(q):
                self.stage = 1
        elif self.stage == 1:
            column = (position - self.start) / self.delta
            if np.linalg.norm(column) < 0.6:
                joint = 1 if self.j == 0 else 3
                self.error = f'M{joint} marker movement too small. Change starting pose/camera view.'
                return False
            self.J[:, self.j] = column
            if self.link.move(self.origin):
                self.stage = 2
        elif self.stage == 2:
            self.j += 1
            if self.j == 2:
                values = np.linalg.svd(self.J, compute_uv=False)
                if values[-1] < 0.5 or values[0] / values[-1] > 25:
                    self.error = 'Poor camera/arm geometry: change pose or camera view.'
                    return False
                return True
            self.stage = 0
        return False

def configuration(args):
    cfg = {'base': 30, 'elbow': 95, 'base_min': 20, 'base_max': 160, 'elbow_min': 65, 'elbow_max': 125, 'offset': [0.0, 0.0]}
    try:
        stored = json.loads(SETTINGS_FILE.read_text())
        for key in cfg:
            if key in stored:
                cfg[key] = stored[key]
    except (OSError, ValueError, TypeError):
        pass
    for key in cfg:
        if key != 'offset':
            value = getattr(args, key, None)
            if value is not None:
                cfg[key] = value
    q = np.array([cfg['base'], cfg['elbow']], float)
    lo = np.array([cfg['base_min'], cfg['elbow_min']], float)
    hi = np.array([cfg['base_max'], cfg['elbow_max']], float)
    offset = np.array(cfg['offset'], float)
    if offset.shape != (2,) or not np.all(np.isfinite(offset)) or (not np.all(np.isfinite([q, lo, hi]))) or np.any(lo < OUTER_LO) or np.any(hi > OUTER_HI) or np.any(q < lo) or np.any(q > hi) or np.any(hi - lo < 6):
        raise ValueError('Invalid saved settings. Run with --reset-settings.')
    SETTINGS_FILE.write_text(json.dumps(cfg, indent=2))
    return (cfg, q, lo, hi, offset)

def overlap_evidence(candidates, previous, age):
    """Image merging is not proof of physical contact."""
    if previous is None or age > 0.8:
        return False
    p0, p1, a0, a1 = previous
    distance = np.linalg.norm(p0 - p1)
    contact_distance = min(160.0, np.sqrt(a0 / np.pi) + np.sqrt(a1 / np.pi) + 22.0)
    if distance > contact_distance:
        return False
    for candidate in candidates:
        if not 0.5 * (a0 + a1) < candidate['area'] < 1.7 * (a0 + a1):
            continue
        near_both = all((cv2.pointPolygonTest(candidate['contour'], tuple(map(float, point)), True) >= -18 for point in (p0, p1)))
        near_midpoint = np.linalg.norm(candidate['p'] - (p0 + p1) / 2) < 65
        if near_both and near_midpoint:
            return True
    return False

class ShoulderFallback:
    """M3-limit assistance; +M2/down was confirmed on this physical setup."""
    def __init__(self, enabled=False, budget=8):
        self.enabled=enabled
        self.budget=budget
        self.reset()

    def reset(self):
        self.used=0
        self.phase=None
        self.message='M2 limit assist ready' if self.enabled else 'M2 limit assist OFF'
        self.clear_history()

    def clear_history(self):
        self.waiting_since=None
        self.waiting_direction=None

    def ready(self, now, target, error, q, lo, hi, planned_target, tolerance, J):
        self.current_tolerance=tolerance
        ey=float(error[1])
        slope=float(J[1,1])  # ORIGINAL calibrated M3 image-Y direction.
        wants_increase=ey*slope>0
        blocked=(q[1]>=hi[1]-.5 if wants_increase else q[1]<=lo[1]+.5)
        if (not self.enabled or self.used>=self.budget or abs(ey)<=tolerance
                or abs(slope)<.2 or not blocked):
            self.clear_history()
            return False
        direction=1 if ey>0 else -1  # User verified ] increases M2 and lowers marker.
        if self.waiting_direction!=direction:
            self.waiting_direction=direction
            self.waiting_since=now
        return now-self.waiting_since>=.35

    def begin(self, link, now, target, offset):
        self.tolerance=self.current_tolerance
        self.goal=np.asarray(target,float).copy()
        self.offset=np.asarray(offset,float).copy()
        self.phase='STOP'
        self.phase_at=now
        self.samples=[]
        self.message='M3 at needed limit: stopping before one slow M2 step'
        self.clear_history()
        link.stop(self.message)

    def cancel(self):
        self.phase=None
        self.clear_history()

    def fail(self, link, message):
        link.stop(message)
        self.message=message+' | holding; F retries'
        self.cancel()
        return 'PAUSE'

    def sampling(self, phase, now):
        self.phase=phase
        self.phase_at=now
        self.samples=[]

    def sample(self, marker, now):
        if now-self.phase_at<.3:
            return None
        self.samples.append(np.asarray(marker,float).copy())
        return np.median(self.samples[-5:],axis=0) if len(self.samples)>=5 else None

    def tick(self, link, valid, pair, now):
        if link.fault:
            return self.fail(link,'M2 communication fault: '+link.fault)
        if not valid:
            return self.fail(link,'M2 assist cancelled: detection lost')
        if self.phase is None or now-self.phase_at>3:
            return self.fail(link,'M2 assist interrupted or timed out')
        marker=pair[0]['p']
        if self.phase in ('MOVE','VERIFY'):
            current_ey=float((pair[1]['p']+self.offset-marker)[1])
            if current_ey*self.direction<=0:
                return self.fail(link,'Target changed vertical side: stopping M2')
        if self.phase=='STOP':
            if not link.stop_waiting:
                self.sampling('BASELINE',now)
        elif self.phase=='BASELINE':
            before=self.sample(marker,now)
            if before is not None:
                self.before=before
                self.goal=np.asarray(pair[1]['p'],float).copy()
                ey=float((self.goal+self.offset-before)[1])
                if abs(ey)<=self.tolerance:
                    return self.fail(link,'Target already within vertical tolerance')
                self.direction=1 if ey>0 else -1
                if not 90<=link.shoulder+self.direction<=165:
                    return self.fail(link,'M2 also reached its allowed limit')
                if link.trim_shoulder(self.direction):
                    self.used+=1
                    self.phase='MOVE'
                    self.phase_at=now
                    self.message='M2: one slow degree '+('down' if self.direction>0 else 'up')
                else:
                    return self.fail(link,'M2 step unavailable')
        elif self.phase=='MOVE':
            if link.shoulder_id is None:
                self.sampling('VERIFY',now)
        elif self.phase=='VERIFY':
            after=self.sample(marker,now)
            if after is not None:
                old_error=self.goal+self.offset-self.before
                new_error=self.goal+self.offset-after
                vertical_gain=abs(old_error[1])-abs(new_error[1])
                total_gain=np.linalg.norm(old_error)-np.linalg.norm(new_error)
                if vertical_gain<.3 or total_gain<.3:
                    return self.fail(link,'M2 step did not measurably help')
                self.message='M2 helped: relearning original M1/M3 then following'
                self.phase=None
                self.clear_history()
                return 'RELEARN'
        return 'RUNNING'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='COM12')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--no-arm', action='store_true')
    parser.add_argument('--no-tap', action='store_true')
    parser.add_argument('--no-shoulder', action='store_true')
    parser.add_argument('--shoulder-start', type=int, required=True)
    parser.add_argument('--shoulder-assist', action='store_true')
    parser.add_argument('--shoulder-budget', type=int, default=8)
    parser.add_argument('--reset-settings', action='store_true')
    for name in ('base', 'elbow', 'base-min', 'base-max', 'elbow-min', 'elbow-max'):
        parser.add_argument('--' + name, type=int)
    args = parser.parse_args()
    if not 90 <= args.shoulder_start <= 165:
        parser.error('--shoulder-start must match the current physical M2 pose within 90-165')
    if not 1 <= args.shoulder_budget <= 75:
        parser.error('--shoulder-budget must be between 1 and 75')
    if args.shoulder_assist and args.no_shoulder:
        parser.error('M2 fallback cannot run with --no-shoulder')
    if args.reset_settings:
        SETTINGS_FILE.unlink(missing_ok=True)
    cfg, start, lo, hi, offset = configuration(args)
    print('Stop diagnostics ON; target movement excluded from divergence checks.')
    print('Original M1/M3 follower and five taps restored. No hand recognition.')
    print('M2:', 'DISABLED; mechanically support it' if args.no_shoulder else f'initial {args.shoulder_start}; 90-165; measured fallback {args.shoulder_assist}')
    print('Starting base/elbow:', start)
    print('Limits:', lo, 'to', hi)
    print('After selecting both objects, a 3-second countdown enables movement.')
    print('M2 starting command must match its current physical angle. SPACE/E stops all motion.')
    print('M4 retains original fixed-90 tracking and five taps. No hand recognition.')
    link = None
    cap = None
    try:
        if not args.no_arm:
            link = Link(args.port, start, shoulder_enabled=not args.no_shoulder, shoulder_start=args.shoulder_start)
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            raise RuntimeError('Camera unavailable. Close other camera apps.')
        vision = Vision()
        cv2.namedWindow(WINDOW)
        clicks = []

        def mouse(event, x, y, flags, parameter):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append((x, y))
        cv2.setMouseCallback(WINDOW, mouse)
        state = 'SELECT'
        message = 'Click GRIPPER marker first, then TARGET'
        J = None
        learner = None
        countdown = None
        lost_at = None
        return_to = 'FOLLOW'
        recover_to = 'FOLLOW'
        sample = None
        worsening = 0
        last_command = 0.0
        enabled_at = 0.0
        tap_enabled = not args.no_tap
        previous = None
        seen_at = -100.0
        merge_frames = 0
        cooldown = 0.0
        escaped_frames = 0
        last_reported_state = None
        fallback = ShoulderFallback(args.shoulder_assist, args.shoulder_budget)
        while True:
            if link:
                link.poll()
            ok, frame = cap.read()
            now = time.monotonic()
            if not ok:
                raise RuntimeError('Camera lost')
            pair, mask = vision.frame(frame)
            while clicks:
                x, y = clicks.pop(0)
                if state in ('SELECT', 'PAUSED', 'ERROR'):
                    vision.click(x, y)
                    message = vision.problem
            valid = pair is not None and vision.good >= 6
            if state == 'SHOULDER':
                outcome = fallback.tick(link, valid, pair, now)
                message = fallback.message
                if outcome == 'PAUSE':
                    state = 'PAUSED'
                    J = None
                elif outcome == 'RELEARN':
                    link.resume()
                    learner = Learner(link, lo, hi)
                    J = None
                    state = 'LEARN'
            elif state != 'FOLLOW':
                fallback.clear_history()
            if valid and state == 'FOLLOW':
                previous = (pair[0]['p'].copy(), pair[1]['p'].copy(), pair[0]['area'], pair[1]['area'])
                seen_at = now
                merge_frames = 0
            if valid:
                lost_at = None
            elif state in ('LEARN', 'FOLLOW', 'ARMING', 'COUNTDOWN', 'HOLD'):
                if lost_at is None:
                    lost_at = now
                if state == 'COUNTDOWN':
                    countdown = None
                    state = 'SELECT'
                    message = 'Waiting for two reliable detections'
                elif state != 'HOLD':
                    return_to = 'NEUTRAL' if state == 'ARMING' else 'LEARN' if state == 'LEARN' else 'FOLLOW'
                    if link:
                        link.stop('Detection lost in ' + state + ': ' + vision.problem)
                    sample = None
                    state = 'HOLD'
                    message = 'Overlap/loss: holding; automatic reacquisition'
                if state == 'HOLD':
                    message = vision.problem + ' | automatic resume when identified'
            if state == 'HOLD' and link and (not link.stop_waiting) and (not valid) and (return_to == 'FOLLOW') and tap_enabled and (now >= cooldown) and overlap_evidence(vision.candidates, previous, now - seen_at):
                merge_frames += 1
                if merge_frames >= 3 and now - link.done_at > 0.1:
                    link.resume()
                    if link.wrist(True):
                        state = 'TAPPING'
                        sample = None
                        message = 'Likely overlap: five M4 tapping cycles'
            else:
                merge_frames = 0
            if state == 'TAPPING' and valid:
                escaped = previous is not None and np.linalg.norm(pair[1]['p'] - previous[1]) > 35
                escaped_frames = escaped_frames + 1 if escaped else 0
                if escaped_frames >= 2:
                    link.stop('Target moved away during tapping')
                    state = 'CANCEL_WAIT'
                    escaped_frames = 0
                    message = 'Target escaped: stopping taps to resume following'
            elif state != 'TAPPING':
                escaped_frames = 0
            if state == 'CANCEL_WAIT' and (not link.stop_waiting) and (not link.fault):
                link.resume()
                recover_to = 'FOLLOW'
                if link.wrist(False):
                    state = 'RECOVER'
                    message = 'Returning wrist to 90, then following target'
            if state == 'TAPPING' and link.wrist_id is None and (not link.fault):
                cooldown = now + 2.0
                previous = None
                seen_at = -100.0
                state = 'HOLD'
                return_to = 'FOLLOW'
                lost_at = now
                vision.good = 0
                valid = False
                message = 'Tapping done. Wrist 90. Searching for both objects.'
            if state == 'RECOVER' and link.wrist_id is None and (not link.fault):
                vision.good = 0
                valid = False
                lost_at = now
                state = 'HOLD'
                return_to = recover_to
                message = 'Wrist returned to 90. Waiting for both objects.'
            if link and link.fault:
                state = 'ERROR'
                message = link.fault
            if state == 'SELECT' and valid and link:
                countdown = now + 3
                state = 'COUNTDOWN'
            if state == 'COUNTDOWN' and valid:
                remaining = max(0, int(np.ceil(countdown - now)))
                message = f'Automatic start in {remaining}s. SPACE cancels.'
                if now >= countdown:
                    if not link.initialized:
                        link.arm()
                        link.initialized = True
                    else:
                        link.resume()
                        link.wrist(False)
                    enabled_at = now
                    state = 'ARMING'
                    message = 'Enabling at starting angles'
            if state == 'ARMING' and valid and link.armed and (link.wrist_id is None) and (now - enabled_at > 0.7):
                learner = Learner(link, lo, hi)
                state = 'LEARN'
                message = 'Learning directions: hold target still'
            if state == 'HOLD' and valid and link and (not link.stop_waiting) and (link.pending is None) and (now - link.done_at > 0.15):
                link.resume()
                if return_to == 'NEUTRAL':
                    recover_to = 'LEARN'
                    if link.wrist(False):
                        state = 'RECOVER'
                        message = 'Returning wrist to neutral'
                elif return_to == 'LEARN' or J is None:
                    learner = Learner(link, lo, hi)
                    state = 'LEARN'
                    message = 'Relearning after interruption'
                else:
                    state = 'FOLLOW'
                    worsening = 0
                    sample = None
                    message = 'Target reacquired: following resumed'
            if state == 'LEARN' and valid:
                if learner.tick(pair[0]['p'], now):
                    J = learner.J.copy()
                    sample = None
                    worsening = 0
                    state = 'FOLLOW'
                    message = 'Directions learned: following automatically'
                    print('Image directions:\n', J)
                elif learner.error:
                    link.stop('Calibration failed: ' + learner.error)
                    state = 'ERROR'
                    message = learner.error
            if state == 'FOLLOW' and valid and (link.pending is None) and (now - link.done_at > 0.05) and (now - last_command > 0.08):
                marker = pair[0]['p']
                error = pair[1]['p'] + offset - marker
                if sample is not None:
                    old_position, old_angles, old_error, old_target = sample
                    dq = link.q - old_angles
                    dp = marker - old_position
                    magnitude_squared = np.dot(dq, dq)
                    if magnitude_squared > 0.8:
                        expected = J @ dq
                        if np.linalg.norm(dp - expected) < max(8, 1.5 * np.linalg.norm(expected)):
                            updated = J + 0.12 * np.outer(dp - expected, dq) / magnitude_squared
                            values = np.linalg.svd(updated, compute_uv=False)
                            if values[-1] > 0.4 and values[0] / values[-1] < 30:
                                J = updated
                        worsening, target_shift, corrected_change = assess_progress(old_position, marker, old_target, pair[1]['p'], offset, worsening)
                        if worsening:
                            print('TRACKING CHECK:', worsening, '/8', 'target shift=%.1fpx' % target_shift, 'fixed-target error change=%+.1fpx' % corrected_change, flush=True)
                    sample = None
                tolerance = max(4, cv2.getTrackbarPos('Match tolerance', 'Green settings'))
                if worsening >= 8:
                    link.stop('Tracking diverged on 8 updates with a stationary target; C relearns')
                    state = 'ERROR'
                    message = 'Tracking diverged: C relearns directions'
                elif np.linalg.norm(error) <= tolerance:
                    message = 'MATCHED: holding'
                else:
                    step = bounded_step(J, error, link.q, lo, hi)
                    target = np.clip(np.rint(link.q + step), lo, hi)
                    if fallback.ready(now, pair[1]['p'], error, link.q, lo, hi, target, tolerance, J):
                        fallback.begin(link, now, pair[1]['p'], offset)
                        state = 'SHOULDER'
                        J = None
                        sample = None
                        message = fallback.message
                    else:
                        if np.max(np.abs(target - link.q)) >= 0.7:
                            if link.move(target):
                                sample = (marker.copy(), link.q.copy(), np.linalg.norm(error), pair[1]['p'].copy())
                                last_command = now
                                message = 'Following target'
                        else:
                            message = 'At joint limit / little useful motion'
            if state != last_reported_state:
                print(time.strftime('[%H:%M:%S]'), 'STATE:', state, '|', message, flush=True)
                last_reported_state = state
            view = frame.copy()
            for candidate in vision.candidates:
                cv2.drawContours(view, [candidate['contour']], -1, (80, 100, 80), 1)
            if pair:
                for role, candidate in enumerate(pair):
                    point = tuple(candidate['p'].astype(int))
                    color = (255, 100, 255) if role == 0 else (0, 255, 255)
                    cv2.drawMarker(view, point, color, cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(view, 'GRIPPER' if role == 0 else 'TARGET', (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                desired = pair[1]['p'] + offset
                cv2.line(view, tuple(pair[0]['p'].astype(int)), tuple(desired.astype(int)), (255, 255, 0), 1)
                error = desired - pair[0]['p']
                cv2.putText(view, f'Error X={error[0]:.0f} Y={error[1]:.0f} px', (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            rows = [state + (' | DETECTION ONLY' if args.no_arm else ''), message, 'SPACE pause | F resume | C relearn | X reselect | Q quit', 'T tapping ' + ('ON' if tap_enabled else 'OFF') + ' | O save gap | Z clear gap']
            if link:
                for joint, name in enumerate(('M1', 'M3')):
                    if link.q[joint] <= lo[joint] + 0.5:
                        flag = ' LOWER LIMIT'
                    elif link.q[joint] >= hi[joint] - 0.5:
                        flag = ' UPPER LIMIT'
                    else:
                        flag = ''
                    rows.append(f'{name} {link.q[joint]:.1f} [{lo[joint]:.0f}..{hi[joint]:.0f}]{flag}')
            if link:
                rows.append(f'M2 {link.shoulder:.1f} [90..165] fallback {"ON" if fallback.enabled else "OFF"} {fallback.used}/{fallback.budget}' if link.shoulder_enabled else 'M2 DISABLED - mechanically supported')
            rows.append('M4: original five taps | M2 assists at the needed M3 limit')
            panel = np.zeros((215, view.shape[1], 3), np.uint8)
            for index, row in enumerate(rows):
                cv2.putText(panel, row, (8, 22 + 23 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 240, 240), 1)
            cv2.imshow(WINDOW, np.vstack([view, panel]))
            cv2.imshow('Green mask', mask)
            key = cv2.waitKey(1) & 255
            if key in (ord('q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord(' '), ord('e')):
                if link:
                    link.stop('Manual pause: Space pressed')
                fallback.cancel()
                state = 'PAUSED'
                sample = None
                countdown = None
                message = 'Paused. F resumes or begins startup.'
            elif key in (ord('['), ord(']')):
                if state == 'PAUSED' and link and link.trim_shoulder(-1 if key == ord('[') else 1):
                    J = None  # Changed geometry: remeasure M1/M3 before following again.
                    learner = None
                    sample = None
                    message = 'M2 manual trim; staying paused. F relearns M1/M3 then follows.'
                else:
                    message = 'M2 trim unavailable: pause, wait for stop/trim acknowledgement, or at limit.'
            elif key == ord('t'):
                tap_enabled = not tap_enabled
                if not tap_enabled and state == 'TAPPING':
                    link.stop('Tapping disabled while wrist moving')
                    state = 'PAUSED'
                    message = 'Tapping stopped. F returns wrist to 90 and resumes.'
            elif key == ord('x'):
                if link:
                    link.stop('Object reselection: X pressed')
                    link.fault = ''
                fallback.reset()
                vision.reset()
                state = 'SELECT'
                J = None
                learner = None
                sample = None
                message = 'Click gripper, then target. Automatic start follows.'
            elif key in (ord('f'), ord('c')) and valid and link and (link.pending is None) and (link.shoulder_id is None) and (not link.stop_waiting) and (state not in ('TAPPING', 'RECOVER', 'CANCEL_WAIT', 'ARMING', 'LEARN', 'SHOULDER')):
                link.fault = ''
                fallback.reset()
                worsening = 0
                sample = None
                if not link.initialized:
                    state = 'SELECT'
                else:
                    link.resume()
                    recover_to = 'LEARN' if key == ord('c') or J is None else 'FOLLOW'
                    if link.wrist(False):
                        state = 'RECOVER'
                        message = 'Returning wrist to neutral before resuming'
            elif key in (ord('f'), ord('c')):
                print('RESUME NOT READY: state=', state, 'detection_valid=', valid, 'stop_ack_pending=', bool(link and link.stop_waiting), '| wait for both objects to be identified, then press F', flush=True)
            elif key in (ord('o'), ord('z')) and state not in ('LEARN', 'ARMING', 'COUNTDOWN', 'TAPPING', 'RECOVER', 'SHOULDER'):
                if key == ord('o') and pair is not None:
                    offset = pair[0]['p'] - pair[1]['p']
                elif key == ord('z'):
                    offset = np.zeros(2)
                else:
                    continue
                cfg['offset'] = offset.tolist()
                SETTINGS_FILE.write_text(json.dumps(cfg, indent=2))
                sample = None
                message = 'Gap saved' if key == ord('o') else 'Gap removed'
    finally:
        if link:
            try:
                link.close()
            except Exception:
                pass
        if cap:
            cap.release()
        cv2.destroyAllWindows()
if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print('\nStopped:', error)
        input('Press Enter to close...')
