MODULE SERVER_R

    !/////////////////////////////////////////////////////////////////////////////////////////////////////////
    !GLOBAL VARIABLES
    !/////////////////////////////////////////////////////////////////////////////////////////////////////////

    !//Robot configuration
    PERS tooldata currentTool:=[TRUE,[[0,0,150],[1,0,0,0]],[0.001,[0,0,0.001],[1,0,0,0],0,0,0]];
    PERS wobjdata currentWobj:=[FALSE,TRUE,"",[[0,0,0],[1,0,0,0]],[[0,0,0],[1,0,0,0]]];
    PERS speeddata currentSpeed;
    PERS zonedata currentZone;

    !//PC communication
    VAR socketdev clientSocket;
    VAR socketdev serverSocket;
    VAR num instructionCode;
    VAR num params{250};
    VAR num nParams;

    PERS string ipController:="192.168.125.1";
    !robot default IP
    !PERS string ipController:= "127.0.0.1"; !local IP for testing in simulation
    VAR num serverPort:=5001;

    !//Motion of the robot
    VAR num collision;
    VAR robtarget cartesianTarget;
    VAR jointtarget jointsTarget;
    VAR bool moveCompleted;
    !Set to true after finishing a Move instruction.

    !//Buffered move variables
    CONST num MAX_BUFFER:=512;
    VAR num BUFFER_POS_C:=0;
    VAR num BUFFER_POS_J:=0;
    VAR robtarget bufferTargets{MAX_BUFFER};
    VAR speeddata bufferSpeeds{MAX_BUFFER};
    VAR jointtarget bufferJointPos{MAX_BUFFER};
    VAR speeddata bufferJointSpeeds{MAX_BUFFER};

    !//External axis position variables
    VAR extjoint externalAxis;

    !//Circular move buffer
    VAR robtarget circPoint;

    !//Correct Instruction Execution and possible return values
    VAR num ok;
    VAR bool should_send_res;
    CONST num SERVER_BAD_MSG:=0;
    CONST num SERVER_OK:=1;

    !//Robot Constants
    CONST jointtarget jposHomeYuMiL:=[[0,-130,30,0,40,0],[-135,9E+09,9E+09,9E+09,9E+09,9E+09]];
    PERS tasks tasklistArms{2}:=[["T_ROB_L"],["T_ROB_R"]];
    VAR syncident Sync_Start_Arms;
    VAR syncident Sync_Stop_Arms;
    PERS num syncReq:=1;
    PERS num syncReqJ:=0;
    !// Change cfx to new version
    CONST confdata R_CONF := [0,0,0,4];
    !//CONST confdata R_CONF := [0,0,0,11];

    !//Camera
    VAR datapos camblock;
    VAR string camname;
    VAR cameradev camdevice;

    !//Added by Chen Hao 2022/01/15 for (CASE 41)
    VAR num path_length :=0;

    !/////////////////////////////////////////////////////////////////////////////////////////////////////////
    !LOCAL METHODS
    !/////////////////////////////////////////////////////////////////////////////////////////////////////////
    !//Method to receive and parse the message received from a PC
    !// If correct message, loads values on:
    !// - instructionCode.
    !// - nParams: Number of received parameters.
    !// - params{nParams}: Vector of received params.
    PROC ReceiveMsg(\num wait_time)
        VAR rawbytes buffer;
        VAR num time_val := WAIT_MAX;  ! default to wait-forever
        VAR num bytes_rcvd; ! parameter received
        IF Present(wait_time) time_val := wait_time;    ! test if wait time is setted

        !//for debug
        !TPErase;
        !TPWrite "START RECEIVING";
        ClearRawBytes buffer;

        !// receive data
        SocketReceive clientSocket, \RawData:=buffer, \ReadNoOfBytes:=1024, \NoRecBytes:=bytes_rcvd, \Time:=time_val;
        !// read number of parameters
        UnpackRawBytes buffer, 1, nParams, \IntX:=UINT;
        !// read instruction code
        UnpackRawBytes buffer, 3, instructionCode, \IntX:=UINT;
        !// parameters are start from 5
        !TPWrite "param no." + NumToStr(nParams,0);
        !TPWrite "instruct no." +NumToStr(instructionCode,3);
        !TPWrite "No of bytes " + NumToStr(bytes_rcvd, 0);
        !TPWrite "No of parameters " + NumToStr((bytes_rcvd-4)/4,5);
        IF (bytes_rcvd-4)/4 <> nParams THEN
            ErrWrite \W, "Socket Recv Failed", "Did not receive expected # of bytes.",
                 \RL2:="Expected: " + ValToStr(nParams),
                 \RL3:="Received: " + ValToStr((bytes_rcvd-3)/4);
             nParams:=-1;
             RETURN;
        ELSE
            !// Read parameters (parameters are defined 4 bytes)
            IF nParams > 0 THEN
                FOR i FROM 1 TO nParams DO
                    UnpackRawBytes buffer, 5 + (i-1)*4, params{i}, \Float4;
                ENDFOR
                !TPWrite NumToStr(instructionCode,0) + " " + NumToStr(params{1},2) + " " + NumToStr(params{2},2)+ " " + NumToStr(params{3},2)+ " " + NumToStr(params{4},2)+ " " + NumToStr(params{5},2)+ " " + NumToStr(params{6},2)+ " " + NumToStr(params{7},2) +" " + NumToStr(params{8},2);
            ENDIF
        ENDIF

        ERROR
            RAISE;  ! raise errors to calling code
    ENDPROC

    !// Added by Chen Hao. This is the function to run MoveJ.
    !// If the motion exec successfully, return True
    !// If the motion exec failed, return False
    FUNC bool ExecMoveJ()
        VAR bool error_flag := FALSE;
        FOR i FROM 1 TO BUFFER_POS_J DO
            IF (i=BUFFER_POS_J) THEN
                MoveAbsJ bufferJointPos{i},bufferJointSpeeds{i},fine,currentTool;
            ELSE
                MoveAbsJ bufferJointPos{i},bufferJointSpeeds{i},z10,currentTool;
            ENDIF
            IF error_flag THEN
                StartMove;
                RETURN FALSE;
            ENDIF
        ENDFOR
        RETURN TRUE;

        ERROR
            TEST ERRNO
            CASE ERR_COLL_STOP:
                TPWrite "ERROR";
                TPWrite NumToStr(ERRNO,0);
                StopMove\Quick;
                ClearPath;
                error_flag := TRUE;
                TRYNEXT;
                !RETURN FALSE;
            DEFAULT:
            ENDTEST
    ENDFUNC

    !//Method to parse the message received from a PC
    !// If correct message, loads values on:
    !// - instructionCode.
    !// - nParams: Number of received parameters.
    !// - params{nParams}: Vector of received params.
    PROC ParseMsg(string msg)
        !//Local variables
        VAR bool auxOk;
        VAR num ind:=1;
        VAR num newInd;
        VAR num length;
        VAR num indParam:=1;
        VAR string subString;
        VAR bool end:=FALSE;

        !//Find the end character
        length:=StrMatch(msg,1,"#");
        IF length>StrLen(msg) THEN
            !//Corrupt message
            nParams:=-1;
        ELSE
            !//Read Instruction code
            newInd:=StrMatch(msg,ind," ")+1;
            subString:=StrPart(msg,ind,newInd-ind-1);
            auxOk:=StrToVal(subString,instructionCode);
            ! ASG: set instructionCode here!
            IF auxOk=FALSE THEN
                !//Impossible to read instruction code
                nParams:=-1;
            ELSE
                ind:=newInd;
                !//Read all instruction parameters (maximum of 8)
                WHILE end=FALSE DO
                    newInd:=StrMatch(msg,ind," ")+1;
                    IF newInd>length THEN
                        end:=TRUE;
                    ELSE
                        subString:=StrPart(msg,ind,newInd-ind-1);
                        auxOk:=StrToVal(subString,params{indParam});
                        indParam:=indParam+1;
                        ind:=newInd;
                    ENDIF
                ENDWHILE
                nParams:=indParam-1;
            ENDIF
        ENDIF
    ENDPROC

    FUNC bool isPoseReachable(robtarget pose, PERS tooldata tool, PERS wobjdata wobj)
        VAR bool reachable := True;
        VAR jointtarget joints;

        joints := CalcJointT(pose, tool \Wobj:=wobj);

        RETURN reachable;

        ERROR
            reachable := FALSE;
            TRYNEXT;
    ENDFUNC

    FUNC bool runContactL(num desired_torque, robtarget pose, PERS zonedata zone, PERS tooldata tool, PERS wobjdata wobj)
        VAR bool is_exe_success := FALSE;
        VAR robtarget current_rbt_pose ;
        VAR robtarget off_rbt_pose ;

        current_rbt_pose:= CRobT(\Tool:=tool\WObj:=wobj);
        off_rbt_pose := Offs(current_rbt_pose,pose.trans.x-current_rbt_pose.trans.x,pose.trans.y-current_rbt_pose.trans.y,pose.trans.z-current_rbt_pose.trans.z);
        TPWrite ValToStr(off_rbt_pose.trans) + " " + ValToStr(off_rbt_pose.rot);
        TPWrite ValToStr(pose.trans) + " " + ValToStr(pose.rot);
        !off_rbt_pose.extax.eax_a := pose.extax.eax_a;
        !MoveL ,v100,zone,tool,\WObj:=wobj;
        !TPWrite ValToStr(current_rbt_pose.trans.x - pose.trans.x) + " " + ValToStr(current_rbt_pose.trans.y - pose.trans.y)+ " " + ValToStr(current_rbt_pose.trans.z-pose.trans.z);
        !ContactL\DesiredTorque:= desired_torque,Offs(current_rbt_pose,pose.trans.x-current_rbt_pose.trans.x,pose.trans.y-current_rbt_pose.trans.y,pose.trans.z-current_rbt_pose.trans.z),v100,\Zone:=zone,tool,\WObj:=wobj;
        ContactL\DesiredTorque:= desired_torque,off_rbt_pose,v100,\Zone:=zone,tool,\WObj:=wobj;
        !ContactL\DesiredTorque:= desired_torque,pose,v100,\Zone:=zone,tool,\WObj:=wobj;
        RETURN is_exe_success;

        ERROR
            is_exe_success := TRUE;
            TRYNEXT;
    ENDFUNC

    FUNC bool isJointsReachable(jointtarget joints, PERS tooldata tool, PERS wobjdata wobj)
        VAR bool reachable := True;
        VAR robtarget pose;

        pose := CalcRobT(joints, tool \Wobj:=wobj);
        cartesianTarget := pose;
        RETURN reachable;

        ERROR
            reachable := FALSE;
            TRYNEXT;
    ENDFUNC

    !//Handshake between server and client:
    !// - Creates socket.
    !// - Waits for incoming TCP connection.
    PROC ServerCreateAndConnect(string ip,num port)
        VAR string clientIP;

        SocketCreate serverSocket;
        SocketBind serverSocket,ip,port;
        SocketListen serverSocket;

        !! ASG: while "current socket status of clientSocket" IS NOT EQUAL TO the "client connected to a remote host"
        WHILE SocketGetStatus(clientSocket)<>SOCKET_CONNECTED DO
            SocketAccept serverSocket,clientSocket\ClientAddress:=clientIP\Time:=WAIT_MAX;
            !//Wait 0.5 seconds for the next reconnection
            WaitTime 0.5;
        ENDWHILE
    ENDPROC

    !//Parameter initialization
    !// Loads default values for
    !// - Tool.
    !// - WorkObject.h
    !// - Zone.
    !// - Speed.
    PROC Initialize()
        currentTool:=[TRUE,[[0,0,0],[1,0,0,0]],[0.001,[0,0,0.001],[1,0,0,0],0,0,0]];
        currentWobj:=[FALSE,TRUE,"",[[0,0,0],[1,0,0,0]],[[0,0,0],[1,0,0,0]]];
        !currentSpeed:=[1500,180,1500,180];
        currentSpeed:=vmax;
        !currentZone:=[FALSE,0.3,0.3,0.3,0.03,0.3,0.03];
        currentZone:=fine; !z0

        !Find the current external axis values so they don't move when we start
        jointsTarget:=CJointT();
        externalAxis:=jointsTarget.extax;

        SetDataSearch "cameradev" \Object:="HandCam_R";
        WHILE GetNextSym(camname,camblock) DO
            GetDataVal camname\Block:=camblock, camdevice;
            CamSetProgramMode camdevice;
            CamLoadJob camdevice, "camright.job";
            CamSetRunMode camdevice;
        ENDWHILE
    ENDPROC

    FUNC string FormateRes(string clientMessage)
        VAR string message;

        message:=NumToStr(instructionCode,0);
        message:=message+" "+NumToStr(ok,0);
        message:=message+" "+ clientMessage;

        RETURN message;
    ENDFUNC

    !/////////////////////////////////////////////////////////////////////////////////////////////////////////
    !//SERVER: Main procedure
    !/////////////////////////////////////////////////////////////////////////////////////////////////////////

    PROC main()
        !//Local variables
        VAR string receivedString;
        !//Received string
        VAR string sendString;
        !//Reply string
        VAR string addString;
        !//String to add to the reply.
        VAR bool connected;
        !//Client connected
        VAR bool reconnected;
        !//Reconnect after sending ack
        VAR bool reconnect;
        !//Drop and reconnection happened during serving a command
        VAR robtarget cartesianPose;
        VAR jointtarget jointsPose;
        !//Create by HAO CHEN (chen960216@gmail.com) 20220114osaka for CASE 70: ContactL
        VAR robtarget pose_tmp;
        VAR extjoint extaxis_tmp;
        VAR jointtarget debug_joint;
        !// Create by HAO CHEN (chen960216@gmail.com) 20230829osaka for update CASE 1 & 5
        VAR confdata T_CONF;

        !//Motion configuration
        ConfL\Off;
        SingArea\Wrist;
        moveCompleted:=TRUE;
        collision:=0;

        !//Initialization of WorkObject, Tool, Speed and Zone
        Initialize;
        !//Socket connection
        connected:=FALSE;
        ServerCreateAndConnect ipController,serverPort;
        connected:=TRUE;
        reconnect:=FALSE;

        !//Server Loop
        WHILE TRUE DO

            !//For message sending post-movement
            should_send_res:=TRUE;

            !//Initialization of program flow variables
            ok:=SERVER_OK;
            !//Has communication dropped after receiving a command?
            addString:="";
            !//Wait for a command
            ReceiveMsg;
            !SocketReceive clientSocket\Str:=receivedString\Time:=WAIT_MAX;
            !TPWrite receivedString;
            !ParseMsg receivedString;

            !//Correctness of executed instruction.
            reconnected:=FALSE;

            !//Execution of the command
            !---------------------------------------------------------------------------------------------------------------
            TEST instructionCode
            CASE 0:
                !Ping
                IF nParams=0 THEN
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 1:
                !Cartesian Move
                !// Revised by Hao CHEN (chen960216@gmail.com) 20230829osaka
                IF nParams=7 OR nParams=11 THEN
                    IF nParams=7 THEN
                        cartesianTarget:=[[params{1},params{2},params{3}],
                                           [params{4},params{5},params{6},params{7}],
                                           R_CONF,
                                           externalAxis];
                    ELSE
                        T_CONF := [params{8}, params{9}, params{10}, params{11}];
                        cartesianTarget:=[[params{1},params{2},params{3}],
                                       [params{4},params{5},params{6},params{7}],
                                       T_CONF,
                                       externalAxis];
                    ENDIF
                    IF isPoseReachable(cartesianTarget, currentTool, currentWobj) THEN
                        ok:=SERVER_OK;
                        moveCompleted:=FALSE;
                        ClkReset clock1;
                        ClkStart clock1;
                        MoveL cartesianTarget,currentSpeed,currentZone,currentTool\WObj:=currentWobj;
                        ClkStop clock1;
                        reg1:=ClkRead(clock1);
                        addString:=NumToStr(reg1,5);
                        moveCompleted:=TRUE;
                    ELSE
                        ok := SERVER_BAD_MSG;
                        addString := "Unreachable Pose";
                    ENDIF
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 2:
                !Joint Move
                !Note the order: eax <--> joint3
                IF nParams=7 THEN
                    externalAxis.eax_a:=params{3};
                    jointsTarget:=[[params{1},params{2},params{4},params{5},params{6},params{7}],externalAxis];
                    ok:=SERVER_OK;
                    moveCompleted:=FALSE;
                    ClkReset clock1;
                    ClkStart clock1;
                    MoveAbsJ jointsTarget,currentSpeed,currentZone,currentTool\Wobj:=currentWobj;
                    ClkStop clock1;
                    reg1:=ClkRead(clock1);
                    addString:=NumToStr(reg1,5);
                    moveCompleted:=TRUE;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 3:
                !Get Cartesian Coordinates (with current tool and workobject)
                IF nParams=0 THEN
                    cartesianPose:=CRobT(\Tool:=currentTool\WObj:=currentWObj);
                    addString:=NumToStr(cartesianPose.trans.x,2)+" ";
                    addString:=addString+NumToStr(cartesianPose.trans.y,2)+" ";
                    addString:=addString+NumToStr(cartesianPose.trans.z,2)+" ";
                    addString:=addString+NumToStr(cartesianPose.rot.q1,3)+" ";
                    addString:=addString+NumToStr(cartesianPose.rot.q2,3)+" ";
                    addString:=addString+NumToStr(cartesianPose.rot.q3,3)+" ";
                    addString:=addString+NumToStr(cartesianPose.rot.q4,3)+" ";
                    !// Added by Hao CHEN (chen960216@gmail.com) 20230829osaka
                    addString:=addString+NumToStr(cartesianPose.robconf.cf1,2)+" ";
                    addString:=addString+NumToStr(cartesianPose.robconf.cf4,2)+" ";
                    addString:=addString+NumToStr(cartesianPose.robconf.cf6,2)+" ";
                    addString:=addString+NumToStr(cartesianPose.robconf.cfx,2);
                    !End of string
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 4:
                !Get Joint Coordinates
                IF nParams=0 THEN
                    jointsPose:=CJointT();
                    addString:=NumToStr(jointsPose.robax.rax_1,2)+" ";
                    addString:=addString+NumToStr(jointsPose.robax.rax_2,2)+" ";
                    !The extra eax is essentially the third axis in our simulator
                    addString:=addString+NumToStr(jointsPose.extax.eax_a,2)+" ";
                    addString:=addString+NumToStr(jointsPose.robax.rax_3,2)+" ";
                    addString:=addString+NumToStr(jointsPose.robax.rax_4,2)+" ";
                    addString:=addString+NumToStr(jointsPose.robax.rax_5,2)+" ";
                    addString:=addString+NumToStr(jointsPose.robax.rax_6,2);
                    !End of string
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 5:
                !Cartesian Move, nonlinear movement
                !// Revised by Hao CHEN (chen960216@gmail.com) 20230829osaka
                IF nParams=7 OR nParams=11 THEN
                    IF nParams=7 THEN
                        T_CONF := [params{8}, params{9}, params{10}, params{11}];
                    ELSE
                        T_CONF.cf1 := params{8};
                        T_CONF.cf4 := params{9};
                        T_CONF.cf6 := params{10};
                        T_CONF.cfx := params{11};
                        cartesianTarget:=[[params{1},params{2},params{3}],
                                       [params{4},params{5},params{6},params{7}],
                                       T_CONF,
                                       externalAxis];
                    ENDIF
                    IF isPoseReachable(cartesianTarget, currentTool, currentWobj) THEN
                        ok:=SERVER_OK;
                        moveCompleted:=FALSE;
                        ClkReset clock1;
                        ClkStart clock1;
                        MoveJ cartesianTarget,currentSpeed,currentZone,currentTool\WObj:=currentWobj;
                        ClkStop clock1;
                        reg1:=ClkRead(clock1);
                        addString:=NumToStr(reg1,5);
                        moveCompleted:=TRUE;
                    ELSE
                        addString := "Unreachable Pose";
                        ok := SERVER_BAD_MSG;
                    ENDIF
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
            CASE 6:
                !Set Tool
                IF nParams=7 THEN
                    currentTool.tframe.trans.x:=params{1};
                    currentTool.tframe.trans.y:=params{2};
                    currentTool.tframe.trans.z:=params{3};
                    currentTool.tframe.rot.q1:=params{4};
                    currentTool.tframe.rot.q2:=params{5};
                    currentTool.tframe.rot.q3:=params{6};
                    currentTool.tframe.rot.q4:=params{7};
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 7:
                !Set Work Object
                IF nParams=7 THEN
                    currentWobj.oframe.trans.x:=params{1};
                    currentWobj.oframe.trans.y:=params{2};
                    currentWobj.oframe.trans.z:=params{3};
                    currentWobj.oframe.rot.q1:=params{4};
                    currentWobj.oframe.rot.q2:=params{5};
                    currentWobj.oframe.rot.q3:=params{6};
                    currentWobj.oframe.rot.q4:=params{7};
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 8:
                !Set Speed of the Robot
                IF nParams=4 THEN
                    currentSpeed.v_tcp:=params{1};
                    currentSpeed.v_ori:=params{2};
                    currentSpeed.v_leax:=params{3};
                    currentSpeed.v_reax:=params{4};
                    ok:=SERVER_OK;
                ELSEIF nParams=2 THEN
                    currentSpeed.v_tcp:=params{1};
                    currentSpeed.v_ori:=params{2};
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 9:
                !Set zone data
                IF nParams=4 THEN
                    IF params{1}=1 THEN
                        currentZone.finep:=TRUE;
                        currentZone.pzone_tcp:=0.0;
                        currentZone.pzone_ori:=0.0;
                        currentZone.zone_ori:=0.0;
                    ELSE
                        currentZone.finep:=FALSE;
                        currentZone.pzone_tcp:=params{2};
                        currentZone.pzone_ori:=params{3};
                        currentZone.zone_ori:=params{4};
                    ENDIF
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 11:
                !Cartesian Move (synchronized)
                IF nParams=7 THEN
                    cartesianTarget:=[[params{1},params{2},params{3}],
                                       [params{4},params{5},params{6},params{7}],
                                       R_CONF,
                                       externalAxis];
                    IF isPoseReachable(cartesianTarget, currentTool, currentWobj) THEN
                        ok:=SERVER_OK;
                        moveCompleted:=FALSE;

                        SyncMoveOn Sync_Start_Arms,tasklistArms;
                        MoveL cartesianTarget\ID:=11,currentSpeed,currentZone,currentTool\WObj:=currentWobj;
                        SyncMoveOff Sync_Stop_Arms;

                        moveCompleted:=TRUE;
                    ELSE
                        addString := "Unreachable Pose";
                        ok := SERVER_BAD_MSG;
                    ENDIF
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 12:
                !Joint Move (synchronized)
                !Note the order: eax <--> joint3
                IF nParams=7 THEN
                    externalAxis.eax_a:=params{3};
                    jointsTarget:=[[params{1},params{2},params{4},params{5},params{6},params{7}],externalAxis];
                    ok:=SERVER_OK;
                    moveCompleted:=FALSE;

                    SyncMoveOn Sync_Start_Arms,tasklistArms;
                    MoveAbsJ jointsTarget\ID:=12,currentSpeed,currentZone,currentTool\Wobj:=currentWobj;
                    SyncMoveOff Sync_Stop_Arms;

                    moveCompleted:=TRUE;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 13:
                !Relative Cartesian Move
                IF nParams=3 THEN
                    cartesianTarget:=Offs(CRobT(),params{1},params{2},params{3});

                    IF isPoseReachable(cartesianTarget, currentTool, currentWobj) THEN
                        ok:=SERVER_OK;
                        moveCompleted:=FALSE;
                        MoveL cartesianTarget,currentSpeed,currentZone,currentTool\WObj:=currentWobj;
                        moveCompleted:=TRUE;
                    ELSE
                        ok := SERVER_BAD_MSG;
                        addString := "Unreachable Pose";
                    ENDIF

                ELSEIF nParams=6 THEN
                    cartesianTarget:=RelTool(CRobT(),params{1},params{2},params{3},\Rx:=params{4}\Ry:=params{5}\Rz:=params{6});

                    IF isPoseReachable(cartesianTarget, currentTool, currentWobj) THEN
                        ok:=SERVER_OK;
                        moveCompleted:=FALSE;
                        MoveL cartesianTarget,currentSpeed,currentZone,currentTool\WObj:=currentWobj;
                        moveCompleted:=TRUE;
                    ELSE
                        ok := SERVER_BAD_MSG;
                        addString := "Unreachable Pose";
                    ENDIF
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 14:
                !ContactL //NOTE: NOT IMPLEMENTED ON PYTHON
                ! Desired Torque Stated
                IF nParams=8 THEN
                    cartesianTarget:=[[params{1},params{2},params{3}],
                                       [params{4},params{5},params{6},params{7}],
                                       R_CONF,
                                       externalAxis];
                    ok:=SERVER_OK;
                    ContactL\DesiredTorque:=params{8},cartesianTarget,v100,\Zone:=currentZone,currentTool,\WObj:=currentWobj;
                    ! Desired Torque Not Stated
                    ! Instruction will only raise the collision detection level and not supervise the internal torque level
                ELSEIF nParams=7 THEN
                    cartesianTarget:=[[params{1},params{2},params{3}],
                                       [params{4},params{5},params{6},params{7}],
                                       R_CONF,
                                       externalAxis];
                    ok:=SERVER_OK;
                    ContactL cartesianTarget,v100,\Zone:=currentZone,currentTool,\WObj:=currentWobj;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 20:
                !Gripper Close
                IF nParams=0 THEN
                    g_GripIn;
                    ok:=SERVER_OK;

                    ! holdForce range = 0 - 20 N, targetPos = 0 - 25 mm, posAllowance = tolerance of gripper closure value
                ELSEIF nParams=2 THEN
                    g_GripIn\holdForce:=params{1}\targetPos:=params{2};
                    ok:=SERVER_OK;

                    ! Program won't wait until gripper completion or failure to move on.
                ELSEIF nParams=3 THEN
                    g_GripIn\holdForce:=params{1}\targetPos:=params{2}\NoWait;
                    ok:=SERVER_OK;

                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 21:
                !Gripper Open
                IF nParams=0 THEN
                    g_GripOut;
                    ok:=SERVER_OK;

                ELSEIF nParams=1 THEN
                    g_GripOut\targetPos:=params{2};
                    ok:=SERVER_OK;

                    ! holdForce range = 0 - 20 N, targetPos = 0 - 25 mm, posAllowance = tolerance of gripper closure value
                ELSEIF nParams=2 THEN
                    g_GripOut\holdForce:=params{1}\targetPos:=params{2};
                    ok:=SERVER_OK;

                    ! Program won't wait until gripper completion or failure to move on.
                ELSEIF nParams=3 THEN
                    g_GripOut\holdForce:=params{1}\targetPos:=params{2}\NoWait;
                    ok:=SERVER_OK;

                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 22:
                ! Initialize gripper with specified values

                ! calibrate only
                IF nParams=0 THEN
                    g_Init\Calibrate;
                    ok:=SERVER_OK;

                    ! set maxSpeed, holdForce, physicalLimit (0-25 mm), and calibrate
                ELSEIF nParams=3 THEN
                    g_Init\maxSpd:=params{1}\holdForce:=params{2}\phyLimit:=params{3}\Calibrate;
                    ok:=SERVER_OK;

                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 23:
                ! Set Max Speed
                IF nParams=1 THEN
                    g_SetMaxSpd params{1};
                    ! between 0-20 mm/s
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 24:
                ! Set gripping force
                IF nParams=0 THEN
                    g_SetForce params{1};
                    ! between 0-20 Newtons
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 25:
                ! Move the gripper to a specified position
                IF nParams=1 THEN
                    g_MoveTo params{1};
                    ! between 0-25 mm or 0-phyLimit if phyLimit is set in CASE 22
                    ok:=SERVER_OK;

                ELSEIF nParams=2 THEN
                    g_MoveTo params{1}\NoWait;
                    ok:=SERVER_OK;

                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 26:
                !Get Gripper Width
                IF nParams=0 THEN
                    addString:=NumToStr(g_GetPos(),2);
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 29:
                ! Stop any action of the gripper (motors will lose power)
                IF nParams=0 THEN
                    g_Stop;
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 30:
                !Add Cartesian Coordinates to buffer
                IF nParams=7 THEN
                    cartesianTarget:=[[params{1},params{2},params{3}],
                                        [params{4},params{5},params{6},params{7}],
                                        R_CONF,
                                        externalAxis];
                    IF isPoseReachable(cartesianTarget, currentTool, currentWobj) THEN
                        IF BUFFER_POS_C<MAX_BUFFER THEN
                            BUFFER_POS_C:=BUFFER_POS_C+1;
                            bufferTargets{BUFFER_POS_C}:=cartesianTarget;
                            bufferSpeeds{BUFFER_POS_C}:=currentSpeed;
                        ENDIF
                    ELSE
                        ok := SERVER_BAD_MSG;
                        addString := "Unreachable Pose";
                    ENDIF

                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 31:
                !Clear Cartesian Buffer
                IF nParams=0 THEN
                    BUFFER_POS_C:=0;
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 32:
                !Get Buffer Size)
                IF nParams=0 THEN
                    addString:=NumToStr(BUFFER_POS_C,2);
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 33:
                !Execute moves in cartesianBuffer as linear moves
                IF nParams=0 THEN
                    FOR i FROM 1 TO (BUFFER_POS_C) DO
                        MoveL bufferTargets{i},bufferSpeeds{i},currentZone,currentTool\WObj:=currentWobj;
                    ENDFOR
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 34:
                !External Axis move
                IF nParams=6 THEN
                    externalAxis:=[params{1},params{2},params{3},params{4},params{5},params{6}];
                    jointsTarget:=CJointT();
                    jointsTarget.extax:=externalAxis;
                    ok:=SERVER_OK;
                    moveCompleted:=FALSE;
                    MoveAbsJ jointsTarget,currentSpeed,currentZone,currentTool\Wobj:=currentWobj;
                    moveCompleted:=TRUE;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 35:
                !Specify circPoint for circular move, and then wait on toPoint
                IF nParams=7 THEN
                    circPoint:=[[params{1},params{2},params{3}],
                                [params{4},params{5},params{6},params{7}],
                                R_CONF,
                                externalAxis];
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 36:
                !specify toPoint, and use circPoint specified previously
                IF nParams=7 THEN
                    cartesianTarget:=[[params{1},params{2},params{3}],
                                        [params{4},params{5},params{6},params{7}],
                                        R_CONF,
                                        externalAxis];
                    MoveC circPoint,cartesianTarget,currentSpeed,currentZone,currentTool\WObj:=currentWobj;
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 37:
                ! AddJointBuffer
                IF nParams=7 THEN
                  jointsTarget:=[[params{1},params{2},params{4},params{5},params{6},params{7}],
                          [params{3},9E9,9E9,9E9,9E9,9E9]];
                  IF BUFFER_POS_J<MAX_BUFFER THEN
                      BUFFER_POS_J:=BUFFER_POS_J+1;
                      bufferJointPos{BUFFER_POS_J}:=jointsTarget;
                      bufferJointSpeeds{BUFFER_POS_J}:=currentSpeed;
                  ENDIF
                  ok:=SERVER_OK;
                ELSE
                  ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
              CASE 38:
                ! ClearJointBuffer
                IF nParams=0 THEN
                  BUFFER_POS_J:=0;
                  !// Added by Chen Hao 2022/01/15
                  path_length :=0;
                  !//
                  ok:=SERVER_OK;
                ELSE
                  ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
              CASE 39:
              ! GetJointBufferSize
                IF nParams=0 THEN
                  addString:=NumToStr(BUFFER_POS_J,2);
                  ok:=SERVER_OK;
                ELSE
                  ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
              CASE 40:
                ! ExecuteJointBuffer
                IF nParams=0 THEN
                  !Trapezoidal velocity
                  !bufferJointSpeeds{1}.v_tcp:=bufferJointSpeeds{1}.v_tcp*0.5;
                  !bufferJointSpeeds{1}.v_ori:=bufferJointSpeeds{1}.v_ori*0.5;
                  !bufferJointSpeeds{2}.v_tcp:=bufferJointSpeeds{2}.v_tcp*0.95;
                  !bufferJointSpeeds{2}.v_ori:=bufferJointSpeeds{2}.v_ori*0.95;
                  !bufferJointSpeeds{BUFFER_POS_J-1}.v_tcp:=bufferJointSpeeds{BUFFER_POS_J-1}.v_tcp*0.95;
                  !bufferJointSpeeds{BUFFER_POS_J-1}.v_ori:=bufferJointSpeeds{BUFFER_POS_J-1}.v_ori*0.95;
                  !bufferJointSpeeds{BUFFER_POS_J}.v_tcp:=bufferJointSpeeds{BUFFER_POS_J}.v_tcp*0.5;
                  !bufferJointSpeeds{BUFFER_POS_J}.v_ori:=bufferJointSpeeds{BUFFER_POS_J}.v_ori*0.5;
                  !Trapezoidal velocity
                    IF ExecMoveJ() THEN
                        addString := "1";
                    ELSE
                        addString := "0";
                    ENDIF
                    ok:=SERVER_OK;
                ELSE
                  ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 41:
                ! Add by Chen Hao 2022/01/15
                IF nParams >= 8 THEN
                    path_length := params{1};
                    FOR i FROM 1 TO path_length DO
                        jointsTarget:=[[params{i*7-5},params{i*7-4},params{i*7-2},params{i*7-1},params{i*7},params{i*7+1}],
                              [params{i*7-3},9E9,9E9,9E9,9E9,9E9]];
                      IF BUFFER_POS_J<MAX_BUFFER THEN
                          BUFFER_POS_J:=BUFFER_POS_J+1;
                          bufferJointPos{BUFFER_POS_J}:=jointsTarget;
                          bufferJointSpeeds{BUFFER_POS_J}:=currentSpeed;
                      ENDIF
                    ENDFOR
                  ok:=SERVER_OK;
                ELSE
                  ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 50:
                ! Request camera image
                IF nParams=0 THEN
                  CamReqImage camdevice;
                  ok:=SERVER_OK;
                ELSE
                  ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 60:
                ! Set vacuum on
                IF nParams=0 THEN
                  g_VacuumOn1;
                  ok:=SERVER_OK;
                ELSE
                  ok:=SERVER_BAD_MSG;
                ENDIF
            CASE 61:
                ! Set vacuum off
                IF nParams=0 THEN
                  g_VacuumOff1;
                  g_BlowOn1;
                  WaitTime 0.1;
                  g_BlowOff1;
                  ok:=SERVER_OK;
                ELSE
                  ok:=SERVER_BAD_MSG;
                ENDIF
            CASE 62:
                ! Get pressure
                IF nParams=0 THEN
                  addString := NumToStr(g_GetPressure1(), 2);
                  ok:=SERVER_OK;
                ELSE
                  ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 70:
                IF nParams=8 THEN
                    ! Test
                    extaxis_tmp.eax_a := params{3};
                    pose_tmp := CalcRobT([[params{1},params{2},params{4},params{5},params{6},params{7}],extaxis_tmp],currentTool \Wobj:=currentWobj);
                    !pose_tmp.robconf := [0,0,1,1];
                    debug_joint := CalcJointT(pose_tmp, currentTool \Wobj:=currentWobj);
                    TPWrite ValToStr(debug_joint.robax) + " " + ValToStr(debug_joint.extax.eax_a);
                    TPWrite ValToStr(pose_tmp.trans);
                    TPWrite ValToStr(pose_tmp.rot);
                    TPWrite NumToStr(params{8},5);
                    IF isPoseReachable(pose_tmp, currentTool, currentWobj) THEN
                        moveCompleted:=FALSE;
                        IF runContactL(params{8}, pose_tmp, currentZone, currentTool, currentWobj) THEN
                            addString :="1";
                            TPWrite "Run Successfully";
                        ELSE
                            addString :="0";
                            TPWrite "Run Error";
                        ENDIF
                        moveCompleted:=TRUE;
                        ok:=SERVER_OK;
                    ENDIF
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 71:
                !// added by Chen Hao 01/23/2022
                IF nParams=0 THEN
                    currentSpeed := vmax;
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
            CASE 95: !// Forward Kinematics, Added by Hao CHEN (chen960216@gmail.com) 20230829osaka
                !returns 1 if given pose is reachable. 0 other wise.
                IF nParams=7 THEN
                    externalAxis.eax_a:=params{3};
                    jointsTarget:=[[params{1},params{2},params{4},params{5},params{6},params{7}],externalAxis];
                    IF isJointsReachable(jointsTarget, currentTool, currentWobj) THEN
                        cartesianPose := CalcRobT(jointsTarget,currentTool \Wobj:=currentWobj);
                        addString:=NumToStr(cartesianPose.trans.x,2)+" ";
                        addString:=addString+NumToStr(cartesianPose.trans.y,2)+" ";
                        addString:=addString+NumToStr(cartesianPose.trans.z,2)+" ";
                        addString:=addString+NumToStr(cartesianPose.rot.q1,3)+" ";
                        addString:=addString+NumToStr(cartesianPose.rot.q2,3)+" ";
                        addString:=addString+NumToStr(cartesianPose.rot.q3,3)+" ";
                        addString:=addString+NumToStr(cartesianPose.rot.q4,3)+" ";
                        addString:=addString+NumToStr(cartesianPose.robconf.cf1,2)+" ";
                        addString:=addString+NumToStr(cartesianPose.robconf.cf4,2)+" ";
                        addString:=addString+NumToStr(cartesianPose.robconf.cf6,2)+" ";
                        addString:=addString+NumToStr(cartesianPose.robconf.cfx,2);
                        !End of string
                        ok:=SERVER_OK;
                    ELSE
                        addString := "Unreachable Pose";
                        ok:=SERVER_BAD_MSG;
                    ENDIF
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 96:
                !returns 1 if given pose is reachable. 0 other wise.
                IF nParams=7 THEN
                    cartesianTarget := [[params{1},params{2},params{3}],
                                       [params{4},params{5},params{6},params{7}],
                                       R_CONF,
                                       externalAxis];
                    IF isPoseReachable(cartesianTarget, currentTool, currentWobj) THEN
                        addString := "1";
                    ELSE
                        addString := "0";
                    ENDIF
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 97:
                !returns 1 if given joint configuration is reachable. 0 other wise.
                IF nParams=7 THEN
                    !// Revised by Hao CHEN (chen960216@gmail.com) 20230829osaka
                    externalAxis.eax_a:=params{3};
                    jointsTarget:=[[params{1},params{2},params{4},params{5},params{6},params{7}],externalAxis];
                    !// ---
                    IF isJointsReachable(jointsTarget, currentTool, currentWobj) THEN
                        addString := "1";
                    ELSE
                        addString := "0";
                    ENDIF
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 98:
                !returns current robot info: serial number, robotware version, and robot type
                IF nParams=0 THEN
                    addString:=GetSysInfo(\SerialNo)+"*";
                    addString:=addString+GetSysInfo(\SWVersion)+"*";
                    addString:=addString+GetSysInfo(\RobotType);
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            CASE 99:
                !Close Connection
                IF nParams=0 THEN
                    reconnect:=TRUE;
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
            CASE 100:
                ! LEFT ARM: Send robot to home
                IF nParams=0 THEN
                    MoveAbsJ jposHomeYuMiL\NoEOffs,currentSpeed,fine,tool0;
                    ok:=SERVER_OK;
                ELSE
                    ok:=SERVER_BAD_MSG;
                ENDIF
                !---------------------------------------------------------------------------------------------------------------
            DEFAULT:
                ok:=SERVER_BAD_MSG;
            ENDTEST
            !---------------------------------------------------------------------------------------------------------------
            !Compose the acknowledge string to send back to the client
            IF connected and reconnected=FALSE and SocketGetStatus(clientSocket)=SOCKET_CONNECTED and should_send_res THEN
                IF reconnect THEN
                    connected:=FALSE;
                    !//Closing the server
                    SocketClose clientSocket;
                    SocketClose serverSocket;
                    !Reinitiate the server
                    ServerCreateAndConnect ipController,serverPort;
                    connected:=TRUE;
                    reconnected:=TRUE;
                    reconnect:=FALSE;
                ELSE
                    SocketSend clientSocket\Str:=FormateRes(addString);
                ENDIF
            ENDIF
        ENDWHILE
ERROR
        ok:=SERVER_BAD_MSG;
        should_send_res:=FALSE;

        TEST ERRNO
            CASE ERR_HAND_WRONGSTATE:
                ok := SERVER_OK;
                RETRY;
            CASE ERR_SOCK_CLOSED:
                connected:=FALSE;
                !//Closing the server
                SocketClose clientSocket;
                SocketClose serverSocket;
                !//Reinitiate the server
                ServerCreateAndConnect ipController,serverPort;
                reconnected:=TRUE;
                connected:=TRUE;
                should_send_res:=TRUE;
                RETRY;

            CASE ERR_HAND_NOTCALIBRATED:
                SocketSend clientSocket\Str:=FormateRes( "ERR_HAND_NOTCALIBRATED: "+NumToStr(ERRNO,0));

                ! Gripper not calibrated.
                g_Init\Calibrate;
                RETRY;

            CASE ERR_COLL_STOP:
                TPWrite "Collision Error R";
                SocketSend clientSocket\Str:=FormateRes("ERR_COLL_STOP: "+NumToStr(ERRNO,0));

                StopMove\Quick;
                ClearPath;
                StorePath;
                !MotionSup\Off;
                cartesianTarget:=Offs(CRobT(),0,0,50);
                MoveL cartesianTarget,v300,fine,currentTool\WObj:=currentWobj;
                !MotionSup\On;
                RestoPath;

                !StartMoveRetry;
                !RETRY;
                !TRYNEXT;

            CASE ERR_ROBLIMIT:
                ! Position is reachable but at least one axis is outside joint limit or limits exceeded for at least one coupled joint (function CalcJoinT)
                SocketSend clientSocket\Str:=FormateRes("ERR_ROBLIMIT: "+NumToStr(ERRNO,0));
                RETRY;

            CASE ERR_OUTSIDE_REACH:
                ! The position (robtarget) is outisde the robot's working area for function CalcJoinT.
                SocketSend clientSocket\Str:=FormateRes("ERR_OUTSIDE_REACH: "+NumToStr(ERRNO,0));
                RETRY;
            DEFAULT:
                SocketSend clientSocket\Str:=FormateRes("Default Error: "+NumToStr(ERRNO,0));
                connected:=FALSE;
                !//Closing the server
                SocketClose clientSocket;
                SocketClose serverSocket;
                !//Reinitiate the server
                ServerCreateAndConnect ipController,serverPort;
                reconnected:=TRUE;
                connected:=TRUE;
                RETRY;
        ENDTEST
    ENDPROC

    
ENDMODULE