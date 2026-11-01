#include <Servo.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Same control and protocol. Following/M2 take ~85% and taps 75% of the preceding quicker version.

// M2: bounded one-degree commands only while stopped; Python measures helpful direction.
Servo baseServo, elbowServo, wristServo, rotationServo, gripServo;
Servo shoulderServo;
// Additive abort latch and slow, externally verified return waypoints.
// Normal tracking/tapping timings below remain unchanged.
bool abortLocked=false, homeBusy=false;
float homeFrom[4], homeTo[4];
unsigned long homeBegan=0;
const unsigned long HOME_DURATION=400;
long homeId=0;

float shoulder=90, shoulderFrom=90, shoulderTo=90;
bool shoulderEnabled=false, shoulderBusy=false;
unsigned long shoulderBegan=0, shoulderDuration=217;
long shoulderId=0;
float b=30, e=95, w=90, fromB, fromE, toB, toE, fromW, toW;
unsigned long began=0, duration=0, lastContact=0, lastStep=0;
bool attached=false, moving=false, stopped=true, wristBusy=false;
long moveId=0, wristId=0;
int wristStage=0;
bool tapping=false;
char line[100]; byte used=0; bool overflowed=false;
const int TAP_POSES[11]={60,105,60,105,60,105,60,105,60,105,90};

void fullPose(const char* tag) {
  Serial.print(tag);Serial.print(' ');Serial.print(b,2);
  Serial.print(' ');Serial.print(shoulder,2);
  Serial.print(' ');Serial.print(e,2);
  Serial.print(' ');Serial.print(w,2);
  Serial.println(F(" 90.00 73.00"));
}
void position(const char* tag) {
  Serial.print(tag); Serial.print(' '); Serial.print(b,2);
  Serial.print(' '); Serial.println(e,2);
  Serial.print(F("SPOS "));Serial.println(shoulder,2);
}
void freeze(const char* tag) {
  moving=false; wristBusy=false; shoulderBusy=false; homeBusy=false; stopped=true; position(tag);
}
void wristSegment(float target) {
  fromW=w; toW=target; began=millis();
  duration=(unsigned long)(abs(toW-w)*7.65f);
  if(duration<64) duration=64;
  wristBusy=true;
}
void command(char* s) {
  lastContact=millis();
  if(!strcmp(s,"K")) return;
  if(!strcmp(s,"I")) {Serial.println(F("BRACCIO_SHOULDER_FALLBACK_V1 GESTURE_LATCH_V1"));return;}
  if(!strcmp(s,"P")) {position("POS");return;}
  if(!strcmp(s,"Q")) {fullPose("POSE");return;}
  if(!strcmp(s,"B")) {abortLocked=true;freeze("STOPPED");return;}
  if(!strcmp(s,"U")) {
    if(moving || wristBusy || shoulderBusy || homeBusy) {Serial.println(F("IGNORED"));return;}
    abortLocked=false;Serial.println(F("UNLOCKED"));return;
  }
  if(!strcmp(s,"S")) {freeze("STOPPED");return;}
  if(!strcmp(s,"R")) {
    if(attached && !shoulderBusy && !homeBusy && !abortLocked){stopped=false;Serial.println(F("RESUMED"));}
    return;
  }
  if(s[0]=='A') {
    int ib,ie,enableShoulder,startShoulder; char extra;
    if(sscanf(s,"A %d %d %d %d %c",&ib,&ie,&enableShoulder,&startShoulder,&extra)!=4 || attached || abortLocked || ib<20 || ib>160 || ie<10 || ie>150 || (enableShoulder!=0 && enableShoulder!=1) || startShoulder<90 || startShoulder>165) {
      Serial.println(F("BAD INIT"));return;
    }
    b=ib;e=ie;w=90;
    shoulderEnabled=(enableShoulder==1);
    if(shoulderEnabled){shoulder=startShoulder;shoulderServo.write(startShoulder);shoulderServo.attach(5);}
    baseServo.write(ib);elbowServo.write(ie);wristServo.write(90);
    rotationServo.write(90);gripServo.write(73);
    baseServo.attach(3);elbowServo.attach(6);wristServo.attach(9);
    rotationServo.attach(10);gripServo.attach(11);
    attached=true;stopped=false;position("ARMED");return;
  }
  // H: <=1 degree, slow waypoint, only under the latched abort stop.
  // Python only replays a matching physically verified path, or an explicit teach jog.
  if(s[0]=='H') {
    if(!attached || !shoulderEnabled || !abortLocked || !stopped || moving || wristBusy || shoulderBusy || homeBusy) {
      Serial.println(F("IGNORED"));return;
    }
    char* token=strtok(s," ");float target[6];char* end;
    for(byte i=0;i<6;i++) {
      token=strtok(NULL," ");
      if(!token){Serial.println(F("BAD COMMAND"));return;}
      target[i]=(float)strtod(token,&end);
      if(*end){Serial.println(F("BAD COMMAND"));return;}
    }
    token=strtok(NULL," ");
    if(!token){Serial.println(F("BAD COMMAND"));return;}
    long id=strtol(token,&end,10);
    if(*end || strtok(NULL," ")){Serial.println(F("BAD COMMAND"));return;}
    const float lower[6]={20,90,10,60,90,73};
    const float upper[6]={160,165,150,105,90,73};
    float current[4]={b,shoulder,e,w};
    for(byte i=0;i<6;i++) {
      if(!(target[i]>=lower[i] && target[i]<=upper[i])){Serial.println(F("LIMIT ERROR"));return;}
      // 0.02 degrees permits the existing two-decimal telemetry rounding.
      if(i<4 && abs(target[i]-current[i])>1.02f){Serial.println(F("LIMIT ERROR"));return;}
    }
    for(byte i=0;i<4;i++){homeFrom[i]=current[i];homeTo[i]=target[i];}
    homeId=id;homeBegan=millis();homeBusy=true;return;
  }
  // J: one bounded manual M2 step while paused; never queued or coordinated with M1/M3.
  if(s[0]=='J') {
    char angleText[24],extra;long id;
    if(sscanf(s,"J %23s %ld %c",angleText,&id,&extra)!=2){Serial.println(F("BAD COMMAND"));return;}
    char* end;float target=(float)strtod(angleText,&end);
    if(*end || !(target>=90 && target<=165) || abs(target-shoulder)>1.001){Serial.println(F("LIMIT ERROR"));return;}
    if(!attached || !shoulderEnabled || !stopped || moving || wristBusy || shoulderBusy || homeBusy || abortLocked){Serial.println(F("IGNORED"));return;}
    shoulderFrom=shoulder;shoulderTo=target;shoulderId=id;
    shoulderBegan=millis();shoulderDuration=217;shoulderBusy=true;return;
  }
  // T = five direct 60-to-105 tapping cycles. N = return wrist to neutral only.
  if(s[0]=='T' || s[0]=='N') {
    char op,extra;long id;
    if(sscanf(s,"%c %ld %c",&op,&id,&extra)!=2) {Serial.println(F("BAD COMMAND"));return;}
    if(!attached || stopped || moving || wristBusy || shoulderBusy || homeBusy || abortLocked) {Serial.println(F("IGNORED"));return;}
    wristId=id;tapping=(op=='T');wristStage=0;
    wristSegment(tapping?TAP_POSES[0]:90);return;
  }
  if(s[0]=='M') {
    int ib,is,ie,iw,ir,ig,d;long id;char extra;
    if(sscanf(s,"M %d %d %d %d %d %d %d %ld %c",&ib,&is,&ie,&iw,&ir,&ig,&d,&id,&extra)!=8) {
      Serial.println(F("BAD COMMAND"));return;
    }
    if(!attached || stopped || moving || wristBusy || shoulderBusy || homeBusy || abortLocked) {Serial.println(F("IGNORED"));return;}
    if(ib<20 || ib>160 || ie<10 || ie>150) {Serial.println(F("LIMIT ERROR"));return;}
    fromB=b;fromE=e;toB=ib;toE=ie;moveId=id;
    float db=abs(toB-b),de=abs(toE-e),change=db>de?db:de;
    duration=(unsigned long)(change*constrain(d,50,200)*.7225f);
    if(duration<72)duration=72;
    began=millis();moving=true;return;
  }
  Serial.println(F("BAD COMMAND"));
}
void setup(){Serial.begin(115200);Serial.println(F("BRACCIO_SHOULDER_FALLBACK_V1 GESTURE_LATCH_V1"));}
void loop(){
  while(Serial.available()) {
    char c=Serial.read();if(c=='\r')continue;
    if(c=='\n') {
      line[used]=0;
      if(overflowed)freeze("BUFFER ERROR");else if(used)command(line);
      used=0;overflowed=false;
    } else if(used<sizeof(line)-1)line[used++]=c;else overflowed=true;
  }
  unsigned long now=millis();
  if(attached && (!stopped || shoulderBusy || homeBusy) && now-lastContact>650)freeze("WATCHDOG");
  if(homeBusy && now-lastStep>=20) {
    lastStep=now;
    float t=(float)(now-homeBegan)/HOME_DURATION;if(t>1)t=1;
    float u=t*t*(3-2*t);
    b=homeFrom[0]+(homeTo[0]-homeFrom[0])*u;
    shoulder=homeFrom[1]+(homeTo[1]-homeFrom[1])*u;
    e=homeFrom[2]+(homeTo[2]-homeFrom[2])*u;
    w=homeFrom[3]+(homeTo[3]-homeFrom[3])*u;
    baseServo.writeMicroseconds((int)(544+b*(2400-544)/180.0));
    shoulderServo.writeMicroseconds((int)(544+shoulder*(2400-544)/180.0));
    elbowServo.writeMicroseconds((int)(544+e*(2400-544)/180.0));
    wristServo.writeMicroseconds((int)(544+w*(2400-544)/180.0));
    if(t>=1) {
      homeBusy=false;
      Serial.print(F("HDONE "));Serial.print(homeId);Serial.print(' ');
      Serial.print(b,2);Serial.print(' ');Serial.print(shoulder,2);Serial.print(' ');
      Serial.print(e,2);Serial.print(' ');Serial.print(w,2);Serial.println(F(" 90.00 73.00"));
    }
  }
  if(shoulderBusy && now-lastStep>=20) {
    lastStep=now;
    float t=(float)(now-shoulderBegan)/shoulderDuration;if(t>1)t=1;
    float u=t*t*(3-2*t);
    shoulder=shoulderFrom+(shoulderTo-shoulderFrom)*u;
    shoulderServo.writeMicroseconds((int)(544+shoulder*(2400-544)/180.0));
    if(t>=1){shoulderBusy=false;Serial.print(F("SDONE "));Serial.print(shoulderId);Serial.print(' ');Serial.println(shoulder,2);}
  }
  if((moving || wristBusy) && now-lastStep>=20) {
    lastStep=now;
    float t=(float)(now-began)/duration;if(t>1)t=1;
    float u=t*t*(3-2*t);
    if(moving) {
      b=fromB+(toB-fromB)*u;e=fromE+(toE-fromE)*u;
      baseServo.writeMicroseconds((int)(544+b*(2400-544)/180.0));
      elbowServo.writeMicroseconds((int)(544+e*(2400-544)/180.0));
      if(t>=1) {
        moving=false;Serial.print(F("DONE "));Serial.print(moveId);
        Serial.print(' ');Serial.print(b,2);Serial.print(' ');Serial.println(e,2);
      }
    } else {
      w=fromW+(toW-fromW)*u;
      wristServo.writeMicroseconds((int)(544+w*(2400-544)/180.0));
      if(t>=1) {
        if(tapping && ++wristStage<11)wristSegment(TAP_POSES[wristStage]);
        else {
          wristBusy=false;tapping=false;
          Serial.print(F("WDONE "));Serial.println(wristId);
        }
      }
    }
  }
}
