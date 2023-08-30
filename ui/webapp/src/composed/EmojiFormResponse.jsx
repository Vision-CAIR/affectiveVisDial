/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React, { useState } from "react";
import {
  FormControl,
  Button,
  Col,
  FormGroup,
  Form
} from "react-bootstrap";


var emojis = [
  {
    id: "00",
    name: "emotion",
    label: "Anger",
    value: "Anger",
    emoji: "https://i.imgur.com/zNDm7kE.jpg",
    isChecked: false,
  },
  {
    id: "01",
    name: "",
    label: "Disgust",
    value: "Disgust",
    emoji: "https://i.imgur.com/yUvJNUW.png",
    isChecked: false,
  },
  {
    id: "02",
    name: "emotion",
    label: "Fear",
    value: "Fear",
    emoji: "https://i.imgur.com/bQsXv2s.png",
    isChecked: false,
  },
  {
    id: "03",
    name: "emotion",
    label: "Sadness",
    value: "Sadness",
    emoji: "https://i.imgur.com/bYLyDAs.png",
    isChecked: false,
  },
  {
    id: "04",
    name: "emotion",
    label: "Excitement",
    value: "Excitement",
    emoji: "https://i.imgur.com/FBMXWUE.png",
    isChecked: false,
  },
  {
    id: "05",
    name: "emotion",
    label: "Amusement",
    value: "Amusement",
    emoji: "https://i.imgur.com/RjnmFuv.png",
    isChecked: false,
  },
  {
    id: "06",
    name: "emotion",
    label: "Contentment",
    value: "Contentment",
    emoji: "https://i.imgur.com/fUoaoi1.png",
    isChecked: false,
  },
  {
    id: "07",
    name: "emotion",
    label: "Awe",
    value: "Awe",
    emoji: "https://i.imgur.com/T93v73O.png",
    isChecked: false,
  },
  {
    id: "08",
    name: "emotion",
    label: "Something Else",
    value: "Something Else",
    emoji: "",
    isChecked: false,
  },
];

const RadioInput = ({ name, label, value, isChecked, handleChange }) => {
  const handleRadioChange = (e) => {
    const { id } = e.currentTarget;
    handleChange(id); // Send back id to radio group for comparison
  };

  return (
    <div>
      {/* Target this input: opacity 0 */}
      <input
        type="radio"
        className="custom-radio"
        name={name}
        id={value} // htlmlFor targets this id.
        checked={isChecked}
        onChange={handleRadioChange}
      />
      <label htmlFor={value}>
        <span>{label}</span>
      </label>
    </div>
  );
};

const RadioGroupInput = () => {
  const [selectedInput, setSelectedInput] = useState("");

  const handleChange = (inputValue) => {
    emojis.forEach((emoji) => {
      if (emoji.id === inputValue) {
        emoji.isChecked = true;
      } else {
        emoji.isChecked = false;
      }
    });
    setSelectedInput(inputValue);
  };

  return (
    <Col sm={11}>
      <div
        style={{
          width: "100%",
          display: "grid",
          gridTemplateColumns: "auto auto auto auto auto auto auto auto auto",
          gridColumnGap: "8x",
          justifyContent: "center",
          alignItems: "center",
          fontSize: "11px"
        }}
      >
        {emojis.map((emoji) => {
          return (
            <div style={{ display: "flex", flexDirection: "column" }}>
              <RadioInput
                key={emoji.id}
                name={emoji.name}
                value={emoji.value}
                label={emoji.label}
                isChecked={emoji.isChecked}
                handleChange={() => handleChange(emoji.id)}
              />
              <img src={emoji.emoji} width="40"></img>
            </div>
          );
        })}
      </div>
    </Col>
  );
};

class EmojiFormResponse extends React.Component {
  // Provide a form-like interface to MTurk interface.

  constructor(props) {
    super(props);
    // At this point it should be assumed that task_data
    // has a field "respond_with_form"
    let responses = [];
    for (let _ of this.props.formOptions) {
      responses.push("");
    }
    this.state = { responses: responses, sending: false };
  }

  tryMessageSend() {
    let form_elements = this.props.formOptions;
    let question = form_elements[0]["question"];
    let response_data = [];
    let response_text = "";
    let all_response_filled = false;

    let answer = "";
    for (let i = 0; i < emojis.length; i++) {
      let e = emojis[i];
      if (e.isChecked){
        answer = e.value;
        all_response_filled = true;
      }
    }
    response_data.push({
      question: question,
      response: answer,
    });
    response_data.push({
      question: "Response Rating",
      response: "rating_value",
    });
    for (let ind in form_elements) {
      let question = form_elements[ind]["question"];
      let response = this.state.responses[ind];
      response_data.push({
        question: question,
        response: response,
      });
      response_text += question + ": " + response + "\n";
    }

    if (!response_data.at(-1).response){
      all_response_filled = false
    }
    if (all_response_filled && this.props.active && !this.state.sending) {
      this.setState({ sending: true });
      this.props
        .onMessageSend({
          text: response_text,
          task_data: { form_responses: response_data },
        })
        .then(() => this.setState({ sending: false }));
      // clear answers once sent
      this.setState((prevState) => {
        prevState["responses"].fill("");
        return { responses: prevState["responses"] };
      });
    }
  }

  render() {
    let form_elements = this.props.formOptions;
    const listFormElements = form_elements.map((form_elem, index) => {
      let question = form_elem["question"];
      let form_type = form_elem["type"];
      if (form_elem["type"] === "choices") {
        let choices = [<option key="empty_option" />].concat(
          form_elem["choices"].map((option_label, index) => {
            return (
              <option key={"option_" + index.toString()}>{option_label}</option>
            );
          })
        );
        return (
          <FormGroup key={"form_el_" + index}>
            <p>{question}</p>
            <RadioGroupInput />
          </FormGroup>
          
        );
      }
      return (
        <FormGroup key={"form_el_" + index}>
          {question}
          
          <Col sm={11}>
            <FormControl
              type="text"
              componentClass="textarea"
              placeholder="at least 10 words"
              style={{ fontSize: "12px", height : '50px' }}
              value={this.state.responses[index]}
              onChange={(e) => {
                var text = e.target.value;
                this.setState((prevState) => {
                  let new_res = prevState["responses"];
                  new_res[index] = text;
                  return { responses: new_res };
                });
              }}
              onKeyPress={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  e.stopPropagation();
                  e.nativeEvent.stopImmediatePropagation();
                }
              }}
            />
          </Col>
          <hr/>
          <hr/>
          {(question === "Why/What makes you feel this particular emotion?") ? "Please, rate the response of the fellow turker" : null}
        </FormGroup>
      );
    });
    let submit_button = (
      <Button
        className="btn btn-primary"
        style={{ height: "30px", width: "100px", fontSize: "12px" }}
        id="id_send_msg_button"
        disabled={
          this.state.textval === "" || !this.props.active || this.state.sending
        }
        onClick={() => this.tryMessageSend()}
      >
        Send
      </Button>
    );

    return (
      <div
        id="response-type-text-input"
        className="response-type-module"
        style={{
          paddingTop: "15px",
          float: "left",
          width: "100%",
          backgroundColor: "#eeeeee",
        }}
      >
        <Form
          horizontal
          style={{ backgroundColor: "#eeeeee", paddingBottom: "2px" }}
        >
          {listFormElements}
          <FormGroup>
            <Col sm={6} />
            <Col sm={5}>{submit_button}</Col>
          </FormGroup>
        </Form>
      </div>
    );
  }
}

export default EmojiFormResponse;