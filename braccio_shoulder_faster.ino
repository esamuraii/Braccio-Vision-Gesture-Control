#include <Servo.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Same control and protocol. Following/M2 take ~85% and taps 75% of the preceding quicker version.

// M2: bounded one-degree commands only while stopped; Python measures helpful direction.
Servo baseServo, elbowServo, wristServo, rotationServo, gripServo;
Servo shoulderServo;
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

void position(const char* tag) {
  Serial.print(tag); Serial.print(' '); Serial.print(b,2);
  Serial.print(' '); Serial.println(e,2);
  Serial.print(F("SPOS "));Serial.println(shoulder,2);
}
void freeze(const char* tag) {
  moving=false; wristBusy=false; shoulderBusy=false; stopped=true; position(tag);
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
  if(!strcmp(s,"I")) {Serial.println(F("BRACCIO_SHOULDER_FALLBACK_V1"));return;}
  if(!strcmp(s,"P")) {position("POS");return;}
  if(!strcmp(s,"S")) {freeze("STOPPED");return;}
  if(!strcmp(s,"R")) {
    if(attached && !shoulderBusy){stopped=false;Serial.println(F("RESUMED"));}
    return;
  }
  if(s[0]=='A') {
    int ib,ie,enableShoulder,startShoulder; char extra;
    if(sscanf(s,"A %d %d %d %d %c",&ib,&ie,&enableShoulder,&startShoulder,&extra)!=4 || attached || ib<20 || ib>160 || ie<10 || ie>150 || (enableShoulder!=0 && enableShoulder!=1) || startShoulder<90 || startShoulder>165) {
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
  // J: one bounded manual M2 step while paused; never queued or coordinated with M1/M3.
  if(s[0]=='J') {
    char angleText[24],extra;long id;
    if(sscanf(s,"J %23s %ld %c",angleText,&id,&extra)!=2){Serial.println(F("BAD COMMAND"));return;}
    char* end;float target=(float)strtod(angleText,&end);
    if(*end || !(target>=90 && target<=165) || abs(target-shoulder)>1.001){Serial.println(F("LIMIT ERROR"));return;}
    if(!attached || !shoulderEnabled || !stopped || moving || wristBusy || shoulderBusy){Serial.println(F("IGNORED"));return;}
    shoulderFrom=shoulder;shoulderTo=target;shoulderId=id;
    shoulderBegan=millis();shoulderDuration=217;shoulderBusy=true;return;
  }
  // T = five direct 60-to-105 tapping cycles. N = return wrist to neutral only.
  if(s[0]=='T' || s[0]=='N') {
    char op,extra;long id;
    if(sscanf(s,"%c %ld %c",&op,&id,&extra)!=2) {Serial.println(F("BAD COMMAND"));return;}
    if(!attached || stopped || moving || wristBusy || shoulderBusy) {Serial.println(F("IGNORED"));return;}
    wristId=id;tapping=(op=='T');wristStage=0;
    wristSegment(tapping?TAP_POSES[0]:90);return;
  }
  if(s[0]=='M') {
    int ib,is,ie,iw,ir,ig,d;long id;char extra;
    if(sscanf(s,"M %d %d %d %d %d %d %d %ld %c",&ib,&is,&ie,&iw,&ir,&ig,&d,&id,&extra)!=8) {
      Serial.println(F("BAD COMMAND"));return;
    }
    if(!attached || stopped || moving || wristBusy || shoulderBusy) {Serial.println(F("IGNORED"));return;}
    if(ib<20 || ib>160 || ie<10 || ie>150) {Serial.println(F("LIMIT ERROR"));return;}
    fromB=b;fromE=e;toB=ib;toE=ie;moveId=id;
    float db=abs(toB-b),de=abs(toE-e),change=db>de?db:de;
    duration=(unsigned long)(change*constrain(d,50,200)*.7225f);
    if(duration<72)duration=72;
    began=millis();moving=true;return;
  }
  Serial.println(F("BAD COMMAND"));
}
void setup(){Serial.begin(115200);Serial.println(F("BRACCIO_SHOULDER_FALLBACK_V1"));}
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
  if(attached && (!stopped || shoulderBusy) && now-lastContact>650)freeze("WATCHDOG");
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
