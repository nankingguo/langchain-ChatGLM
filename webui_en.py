import gradio as gr
import os
import shutil
from chains.local_doc_qa import LocalDocQA
from configs.model_config import *
import nltk

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

embedding_model_dict_list = list(embedding_model_dict.keys())

llm_model_dict_list = list(llm_model_dict.keys())

local_doc_qa = LocalDocQA()

flag_csv_logger = gr.CSVLogger()

def get_vs_list():
    lst_default = ["Create New Knowledge Base"]
    lst = local_doc_qa.get_collections()
    if not lst:
        return lst_default
    lst.sort()
    return lst + lst_default


vs_list = get_vs_list()


def get_answer(query, vs_path, history, mode,
               streaming: bool = STREAMING):
                   if mode == "Knowledge Base Q&A" and vs_path:
                       for resp, history in local_doc_qa.get_knowledge_based_answer(
                           query=query,
                           vs_path=vs_path,
                           chat_history=history,
                           streaming=streaming):
                               source = "\n\n"
                               source += "".join(
                                   [f"""<details> <summary>Source [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
                                    f"""{doc.page_content}\n"""
                                    f"""</details>"""
                                    for i, doc in
                                    enumerate(resp["source_documents"])])
                               history[-1][-1] += source
                               yield history, ""
                   else:
                       for resp, history in local_doc_qa.llm._call(query, history,
                                                                   streaming=streaming):
                                                                       history[-1][-1] = resp + (
                                                                           "\n\nThe current knowledge base is empty. To perform Q&A based on the knowledge base, please load the knowledge base first before asking questions。" if mode == "Knowledge Base Q&A" else "")
                                                                       yield history, ""
                   logger.info(f"flagging: username={FLAG_USER_NAME},query={query},vs_path={vs_path},mode={mode},history={history}")
                   flag_csv_logger.flag([query, vs_path, history, mode], username=FLAG_USER_NAME)

def init_model():
    try:
        local_doc_qa.init_cfg()
        local_doc_qa.llm._call("Hello")
        reply = """The model has been successfully loaded, you can start the conversation, or select the mode from the right to start the conversation"""
        logger.info(reply)
        return reply
    except Exception as e:
        logger.error(e)
        reply = """The model was not successfully loaded. Please reselect it in the "Model Configuration" tab in the upper left corner of the page and click the "Load Model" button"""
        if str(e) == "Unknown platform: darwin":
            logger.info("This error may be due to the fact that you are using the macOS operating system and need to download the model locally before executing the web UI. For specific methods, please refer to the local deployment method and common problems in the README："
                        " https://github.com/imClumsyPanda/langchain-ChatGLM")
        else:
            logger.info(reply)
        return reply


def reinit_model(llm_model, embedding_model, llm_history_len, use_ptuning_v2, use_lora, top_k, history):
    try:
        local_doc_qa.init_cfg(llm_model=llm_model,
                              embedding_model=embedding_model,
                              llm_history_len=llm_history_len,
                              use_ptuning_v2=use_ptuning_v2,
                              use_lora=use_lora,
                              top_k=top_k, )
        model_status = """The model has been successfully reloaded, you can start the conversation, or select the mode from the right to start the conversation"""
        logger.info(model_status)
    except Exception as e:
        logger.error(e)
        model_status = """The model was not successfully reloaded. Please go to the "Model Configuration" tab in the upper left corner of the page and select again, then click the "Load Model" button"""
        logger.info(model_status)
    return history + [[None, model_status]]


def get_vector_store(vs_id, files, history):
    vs_path = vs_id
    filelist = []
    if not os.path.exists(os.path.join(UPLOAD_ROOT_PATH, vs_id)):
        os.makedirs(os.path.join(UPLOAD_ROOT_PATH, vs_id))
    for file in files:
        filename = os.path.split(file.name)[-1]
        shutil.move(file.name, os.path.join(UPLOAD_ROOT_PATH, vs_id, filename))
        filelist.append(os.path.join(UPLOAD_ROOT_PATH, vs_id, filename))
    if local_doc_qa.llm and local_doc_qa.embeddings:
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, vs_path)
        if len(loaded_files):
            file_status = f"Uploaded {'、'.join([os.path.split(i)[-1] for i in loaded_files])} to Knowledge Base，please start asking questions"
        else:
            file_status = "The file was not successfully loaded. Please upload the file again"
    else:
        file_status = "The model has not completed loading. Please load the model before importing the file"
        vs_path = None
    logger.info(file_status)
    return vs_path, None, history + [[None, file_status]]


def change_vs_name_input(vs_id, history):
    if vs_id == "Create New Knowledge Base":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), None, history
    else:
        file_status = f"Uploaded Knowledge Base {vs_id}，please start asking questions"
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), vs_id, history + [
            [None, file_status]]


def change_mode(mode):
    if mode == "Knowledge Base Q&A":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def add_vs_name(vs_name, vs_list, chatbot):
    if vs_name in vs_list:
        vs_status = "Conflicting with existing knowledge base name. Please reselect a different name and submit"
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True), vs_list,gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),  chatbot
    else:
        vs_status = f"""Uploaded Knowledge Base"{vs_name}",After uploading the file and successfully loading it, it will be stored. Please complete the file upload before starting the conversation. """
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True, choices= [vs_name] + vs_list, value=vs_name), [vs_name]+vs_list, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),chatbot

block_css = """.importantButton {
            background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
                                        border: none !important;
                                        }
                                        .importantButton:hover {
            background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
                                        border: none !important;
                                        }"""

webui_title = """
                                        # ChatGLM-6B + AnalyticDB For PostgreSQL(Vector version) + langchain WebUI
                                        """
default_vs = vs_list[0] if len(vs_list) > 0 else "is null"
init_message = f"""Welcome langchain+AnalyticDB+ChatGLM Web UI！
<p><strong><font color="red">【Special Notice:
The big model used in this demo is open-source THUDM/ChatGLM-6B, with detailed reference to: https://github.com/THUDM/ChatGLM-6B For learning purposes only.
The model is based on ChatGLM-6B, developed by a team from Tsinghua University. It is an open source dialogue language model that supports both Chinese and English. It is based on the General Language Model (GLM) architecture and has 6.2 billion parameters.
This project is only for user scientific research, please consciously comply with it https://huggingface.co/THUDM/chatglm-6b/blob/main/MODEL_LICENSE Protocol.
Alibaba Cloud does not guarantee the legality, security, and accuracy of the third-party model you use on the image, and is not responsible for any damage caused thereby;
You should consciously abide by the user agreement, usage specifications, and relevant laws and regulations of the third-party model installed on the image, and bear relevant responsibilities for the legality and compliance of using the third-party model on your own.】</font></strong></p>
Please switch modes on the right side. Currently, it supports direct conversations with LLM models or Q&A based on local knowledge bases.
Knowledge base Q&A mode. After selecting the name of the knowledge base, you can start Q&A. The current knowledge base {default_vs} can be uploaded to the knowledge base after selecting the name if necessary.
"""

hits_msg = """
## 【Special Notice：
## 1）The big model used in this demo is open-source THUDM/ChatGLM-6B, with detailed reference to: https://github.com/THUDM/ChatGLM-6B For learning purposes only.
## 2）The model is based on ChatGLM-6B, developed by a team from Tsinghua University. It is an open source dialogue language model that supports both Chinese and English. It is based on the General Language Model (GLM) architecture and has 6.2 billion parameters.
## 3）This project is only for user scientific research, please consciously comply with it https://huggingface.co/THUDM/chatglm-6b/blob/main/MODEL_LICENSE Protocol.
## 4）Alibaba Cloud does not guarantee the legality, security, and accuracy of the third-party model you use on the image, and is not responsible for any damage caused thereby;
## You should consciously abide by the user agreement, usage specifications, and relevant laws and regulations of the third-party model installed on the image, and bear relevant responsibilities for the legality and compliance of using the third-party model on your own.】"""

model_status = init_model()
default_path = vs_list[0] if len(vs_list) > 0 else ""

with gr.Blocks(css=block_css) as demo:
    vs_path, file_status, model_status, vs_list = gr.State(default_path), gr.State(""), gr.State(
        model_status), gr.State(vs_list)

    gr.Markdown(webui_title)
    gr.Markdown(hits_msg)
    with gr.Tab("conversation"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, init_message], [None, model_status.value]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)
                query = gr.Textbox(show_label=False,
                                   placeholder="Please input the question content and press Enter to submit").style(container=False)
            with gr.Column(scale=5):
                mode = gr.Radio(["LLM conversation", "Knowledge base Q&A"],
                                label="Please select the mode",
                                value="Knowledge base Q&A", )
                vs_setting = gr.Accordion("Configure Knowledge Base")
                mode.change(fn=change_mode,
                            inputs=mode,
                            outputs=vs_setting)
                with vs_setting:
                    select_vs = gr.Dropdown(vs_list.value,
                                            label="Please select the knowledge base to load",
                                            interactive=True,
                                            value=vs_list.value[0] if len(vs_list.value) > 0 else None
                                            )
                    vs_name = gr.Textbox(label="Please enter the name of the new knowledge base",
                                         lines=1,
                                         interactive=True,
                                         visible=True if default_path=="" else False)
                    vs_add = gr.Button(value="Add to Knowledge Base Options",  visible=True if default_path=="" else False)                   
                    file2vs = gr.Column(visible=False if default_path=="" else True)
                    with file2vs:
                        # load_vs = gr.Button("Load Knowledge Base")
                        gr.Markdown("Add files to the knowledge base")
                        with gr.Tab("Upload files"):
                            files = gr.File(label="Add Files",
                                            file_types=['.txt', '.md', '.docx', '.pdf'],
                                            file_count="multiple",
                                            show_label=False
                                            )
                            load_file_button = gr.Button("Upload files and load the knowledge base")
                        with gr.Tab("Upload folder"):
                            folder_files = gr.File(label="Add Files",
                                                   # file_types=['.txt', '.md', '.docx', '.pdf'],
                                                   file_count="directory",
                                                   show_label=False
                                                   )
                            load_folder_button = gr.Button("Upload folder and load knowledge base")
                    # load_vs.click(fn=)
                    vs_add.click(fn=add_vs_name,
                                 inputs=[vs_name, vs_list, chatbot],
                                 outputs=[select_vs, vs_list,vs_name,vs_add, file2vs,chatbot])
                    select_vs.change(fn=change_vs_name_input,
                                     inputs=[select_vs, chatbot],
                                     outputs=[vs_name, vs_add, file2vs, vs_path, chatbot])
                    # Save the uploaded file to the content folder and update the dropdown box
                    load_file_button.click(get_vector_store,
                                           show_progress=True,
                                           inputs=[select_vs, files, chatbot],
                                           outputs=[vs_path, files, chatbot],
                                           )
                    load_folder_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs, folder_files, chatbot],
                                             outputs=[vs_path, folder_files, chatbot],
                                             )
                    flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
                    query.submit(get_answer,
                                 [query, vs_path, chatbot, mode],
                                 [chatbot, query])
    with gr.Tab("Model configuration"):
        llm_model = gr.Radio(llm_model_dict_list,
                             label="LLM model",
                             value=LLM_MODEL,
                             interactive=True)
        llm_history_len = gr.Slider(0,
                                    10,
                                    value=LLM_HISTORY_LEN,
                                    step=1,
                                    label="The Number of LLM conversation rounds",
                                    interactive=True)
        use_ptuning_v2 = gr.Checkbox(USE_PTUNING_V2,
                                     label="Model tuned using p-tuning v2",
                                     interactive=True)
        use_lora = gr.Checkbox(USE_LORA,
                               label="Weighting with Lora fine-tuning",
                               interactive=True)
        embedding_model = gr.Radio(embedding_model_dict_list,
                                   label="Embedding model",
                                   value=EMBEDDING_MODEL,
                                   interactive=True)
        top_k = gr.Slider(1,
                          20,
                          value=VECTOR_SEARCH_TOP_K,
                          step=1,
                          label="Vector matching top k",
                          interactive=True)
        load_model_button = gr.Button("Reload Model")
    load_model_button.click(reinit_model,
                            show_progress=True,
                            inputs=[llm_model, embedding_model, llm_history_len, use_ptuning_v2, use_lora, top_k,
                                    chatbot],
                            outputs=chatbot
                            )

(demo
 .queue(concurrency_count=3)
 .launch(server_name='0.0.0.0',
         server_port=7860,
         show_api=False,
         share=False,
         inbrowser=False))
