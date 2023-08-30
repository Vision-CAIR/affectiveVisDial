/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import ReactDOM from "react-dom";
import "bootstrap-chat/styles.css";
import "./css/custom-styles.css";

import { AffectiveChatApp } from "./composed/AffectiveChatApp.jsx";

import { DefaultTaskDescription } from "bootstrap-chat";

function PrintCaptionData({taskContext}){
  return ((taskContext.positive_emoji_url) ? 
 (
    <div className="row">
      <div className="col-lg-6">
        <div
            style={{
              width: 70,
              height: 70,
            }}
        >
            <img
              src={taskContext.negative_emoji_url}
              alt="Emoji-icon1" 
              style={{ width: "100%", height: "100%" }}
            />
        </div>
        <h4>{taskContext.negative_emotion_label}</h4>
        <h5>{taskContext.negative_caption}</h5>
      </div>

      <div className="col-lg-6">
        <div
              style={{
                width: 70,
                height: 70,
              }}
          >
            <img
              src={taskContext.positive_emoji_url}
              alt="Emoji-icon2" 
              style={{ width: "100%", height: "100%" }}
            />
        </div>
        <h4>{taskContext.positive_emotion_label}</h4>
        <h5>{taskContext.positive_caption}</h5>
      </div>
    </div>
  ) :  null);
}

function PrintImage({img_src}){
  return ((img_src) ?
   (
    <div className="row">
    <div className="col-md-8 offset-md-2">
      
      <img
        src={img_src}
        className="mx-auto d-block"
        alt="Image"
      />
    </div>
  </div>
  ) : null);
}

function ChatMessage({ isSelf, idx, agentName, message = "", onRadioChange }) {
  const floatToSide = isSelf ? "right" : "left";
  const alertStyle = isSelf ? "alert-info" : "alert-warning";
  const handleChange = (e) => {
    onRadioChange(e.currentTarget.value);
  };

  return (
    <div className="row" style={{ marginLeft: "0", marginRight: "0" }}>
      <div
        className={"alert message " + alertStyle}
        role="alert"
        style={{ float: floatToSide }}
      >
        <span style={{ fontSize: "16px", whiteSpace: "pre-wrap" }}>
          {agentName === "Chat Agent 1" && (
            <div
              style={{
                width: "100%",
                display: "flex",
                flexDirection: "row",
                justifyContent: "flex-start",
                alignItems: "center",
              }}
            >
              <div
                style={{
                  width: 100,
                  height: 100,
                }}
              >
                <img
                  src="https://affective-dialog.s3.us-west-2.amazonaws.com/assets/Questioner_icon_with_title.png"
                  alt="Questioner-icon"
                  style={{ width: "100%", height: "100%" }}
                />
              </div>
              <span></span>
            </div>
          )}
          {agentName === "Chat Agent 2" && (
            <div
              style={{
                width: "100%",
                display: "flex",
                flexDirection: "row",
                justifyContent: "flex-start",
                alignItems: "center",
              }}
            >
              <div
                style={{
                  width: 100,
                  height: 100,
                }}
              >
                <img
                  src="https://affective-dialog.s3.us-west-2.amazonaws.com/assets/Answerer_icon_with_title.png" 
                  alt="Answerer-icon" 
                  style={{ width: "100%", height: "100%" }}
                />
              </div>
              <span></span>
            </div>
          )}
          <span style={{ fontSize: "16px", whiteSpace: "pre-wrap" }} dangerouslySetInnerHTML={{ __html: message }}>
        </span>
        </span>
      </div>
    </div>
  );
}

function RenderChatMessage({ message, mephistoContext, appContext, idx }) {
  const { agentId } = mephistoContext;
  const { currentAgentNames } = appContext.taskContext;

  return (
    <div onClick={() => alert("You clicked on message with index " + idx)}>
      <ChatMessage
        isSelf={message.id === agentId || message.id in currentAgentNames}
        agentName={
          message.id in currentAgentNames
            ? currentAgentNames[message.id]
            : message.id
        }
        message={message.text}
        taskData={message.task_data}
        messageId={message.update_id}
      />
    </div>
  );
}

function MainApp() {
  return (
    <AffectiveChatApp
    renderMessage={({ message, idx, mephistoContext, appContext }) => (
      <RenderChatMessage
        message={message}
        mephistoContext={mephistoContext}
        appContext={appContext}
        idx={idx}
        key={message.message_id + "-" + idx}
      >
      </RenderChatMessage>
    )}
    renderSidePane={({
      mephistoContext: { taskConfig, agentId },
      appContext: { taskContext },
    }) => (
      <DefaultTaskDescription
        chatTitle={taskConfig.chat_title}
        taskDescriptionHtml={taskConfig.task_description}
      >
        <br/>
        <hr/>
        {(agentId.includes("onboarding")) ? null : <PrintCaptionData taskContext={taskContext}></PrintCaptionData>}
        {(agentId.includes("onboarding")) ? null : <PrintImage img_src={taskContext.image_src}></PrintImage>}

      </DefaultTaskDescription>
      )}
    />
  );
}

ReactDOM.render(<MainApp />, document.getElementById("app"));
