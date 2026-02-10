from langchain.agents import create_agent
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from .logging import log_msg
from .prompts import (
    SYS_ROUTER,
    SYS_QUIZ_DETECTOR,
    SYS_SYLLABUS_EXTRACTOR,
    SYS_ASSIGNMENT_ANALYZER,
    SYS_DIGEST,
    SYS_MATERIAL_SUMMARIZER,
    SYS_DOC_WRITER,
    SYS_OUTLINE,
    SYS_STEP_EXPLAINER,
    SYS_DOC_FORMATTER,
    SYS_QUIZ_TOPIC,
    SYS_QUIZ_QA,
    SYS_CODEGEN,
)


def make_json_agent(name: str, model, tools, system_text: str, debug: bool = False):
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_text.strip(),
        debug=debug,
        name=name,
    )


def build_agents(model_name: str, tools: dict):
    llm = ChatNVIDIA(model=model_name, temperature=0.2)

    router_agent = make_json_agent(
        name="router_agent",
        model=llm,
        tools=[tools["db_get_event"]],
        system_text=SYS_ROUTER,
        debug=False,
    )

    quiz_detector_agent = make_json_agent(
        name="quiz_detector_agent",
        model=llm,
        tools=[tools["db_get_event"]],
        system_text=SYS_QUIZ_DETECTOR,
        debug=False,
    )

    syllabus_extractor_agent = make_json_agent(
        name="syllabus_extractor_agent",
        model=llm,
        tools=[tools["db_get_event"], tools["db_get_recent_events"], tools["notify_whatsapp"]],
        system_text=SYS_SYLLABUS_EXTRACTOR,
        debug=False,
    )

    assignment_analyzer_agent = make_json_agent(
        name="assignment_analyzer_agent",
        model=llm,
        tools=[tools["db_get_event"]],
        system_text=SYS_ASSIGNMENT_ANALYZER,
        debug=False,
    )

    digest_agent = make_json_agent(
        name="digest_agent",
        model=llm,
        tools=[tools["db_get_recent_events"], tools["notify_whatsapp"]],
        system_text=SYS_DIGEST,
        debug=False,
    )

    material_summarizer_agent = make_json_agent(
        name="material_summarizer_agent",
        model=llm,
        tools=[],
        system_text=SYS_MATERIAL_SUMMARIZER,
        debug=False,
    )

    doc_writer_agent = make_json_agent(
        name="doc_writer_agent",
        model=llm,
        tools=[tools["db_get_event"], tools["docs_create"], tools["docs_append"], tools["notify_whatsapp"]],
        system_text=SYS_DOC_WRITER,
        debug=False,
    )

    outline_agent = make_json_agent(
        name="outline_agent",
        model=llm,
        tools=[],
        system_text=SYS_OUTLINE,
        debug=False,
    )

    step_explainer_agent = make_json_agent(
        name="step_explainer_agent",
        model=llm,
        tools=[],
        system_text=SYS_STEP_EXPLAINER,
        debug=False,
    )

    formatter_agent = make_json_agent(
        name="formatter_agent",
        model=llm,
        tools=[],
        system_text=SYS_DOC_FORMATTER,
        debug=False,
    )

    quiz_topic_agent = make_json_agent(
        name="quiz_topic_agent",
        model=llm,
        tools=[],
        system_text=SYS_QUIZ_TOPIC,
        debug=False,
    )

    quiz_qa_agent = make_json_agent(
        name="quiz_qa_agent",
        model=llm,
        tools=[],
        system_text=SYS_QUIZ_QA,
        debug=False,
    )

    codegen_agent = make_json_agent(
        name="codegen_agent",
        model=llm,
        tools=[],
        system_text=SYS_CODEGEN,
        debug=False,
    )

    log_msg(
        "agents.created agents="
        + ", ".join(
            [
                "router_agent",
                "quiz_detector_agent",
                "syllabus_extractor_agent",
                "assignment_analyzer_agent",
                "digest_agent",
                "material_summarizer_agent",
                "doc_writer_agent",
                "outline_agent",
                "step_explainer_agent",
                "formatter_agent",
                "quiz_topic_agent",
                "quiz_qa_agent",
                "codegen_agent",
            ]
        )
    )

    return {
        "llm": llm,
        "router_agent": router_agent,
        "quiz_detector_agent": quiz_detector_agent,
        "syllabus_extractor_agent": syllabus_extractor_agent,
        "assignment_analyzer_agent": assignment_analyzer_agent,
        "digest_agent": digest_agent,
        "material_summarizer_agent": material_summarizer_agent,
        "doc_writer_agent": doc_writer_agent,
        "outline_agent": outline_agent,
        "step_explainer_agent": step_explainer_agent,
        "formatter_agent": formatter_agent,
        "quiz_topic_agent": quiz_topic_agent,
        "quiz_qa_agent": quiz_qa_agent,
        "codegen_agent": codegen_agent,
    }
