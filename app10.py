import streamlit as st
import time
import pandas as pd
import json
import os
import random
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# langchain related imports
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException
from pydantic import BaseModel, Field

# streamlit cache related import
from functools import lru_cache

# Pydantic models for structured output
class BloomClassification(BaseModel):
    bloom_level: str = Field(description="The Bloom's taxonomy level: 기억, 이해, 적용, 분석, 평가, or 창조")

class QuestionSuggestion(BaseModel):
    suggested_question: str = Field(description="A single suggested question in Korean ending with a question mark")

# Import paragraphs from config file
try:
    from paragraphs_config import get_paragraphs
except ImportError:
    # Fallback if config file doesn't exist
    def get_paragraphs(count=45):
        return [f"Sample paragraph {i+1}" for i in range(count)]

# Load environment variables (for OpenAI API key)
load_dotenv()

# Check if OpenAI API key is available
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: No OpenAI API key found in environment variables.")
    print("Please set OPENAI_API_KEY in your .env file or environment variables.")
else:
    print("OpenAI API key loaded successfully.")

# Few-shot examples for Bloom's taxonomy classification
BLOOM_CLASSIFICATION_EXAMPLES = [
    {
        "paragraph": "8세기 후반 바그다드에는 '지혜의 집(Bayt al-Hikma)'이라는 지식 집약 기관이 설립되어, 고대 그리스의 철학과 자연과학 문헌을 아랍어로 번역하는 대규모 작업이 이루어졌다. 이 번역은 단순히 언어를 바꾸는 것이 아니라, 플라톤, 아리스토텔레스, 히포크라테스 등의 사상을 해석하고 보완하며 새로운 학문 체계를 세우는 과정이었다.",
        "question": "지혜의 집은 언제 설립되었을까?",
        "bloom_level": "기억"
    },
    {
        "paragraph": "현대 도시계획에서 녹지 공간의 중요성이 대두되고 있다. 녹지는 대기 정화, 온도 조절, 시민의 정신 건강 향상 등 다양한 기능을 수행한다. 특히 팬데믹 이후 야외 활동 공간에 대한 수요가 급증하면서, 도시 내 공원과 정원의 역할이 재조명받고 있다.",
        "question": "녹지 공간이 시민들에게 제공하는 주요 혜택들은 무엇일까?",
        "bloom_level": "이해"
    },
    {
        "paragraph": "예측 처리 이론(predictive processing theory)은 인간의 뇌가 외부 자극을 받아들이기만 하는 수동적 기관이 아니라, 끊임없이 미래의 감각 정보를 예측하고 그 예측이 실제 감각 정보와 얼마나 일치하는지를 비교하면서 작동한다고 설명한다. 즉, 뇌는 예상과 다른 정보가 들어올 때 그 오류를 수정해나가는 방식으로 작동한다. 예측이 잘 맞으면 뇌의 에너지 사용은 줄어들고, 오류가 있으면 더 많은 자원이 동원되어 환경에 대한 새로운 모델을 학습하게 된다. 이 이론은 ‘정보가 입력되고 처리된다’는 고전적 인지 이론과 달리, 뇌가 능동적으로 세계를 구성한다는 관점의 전환을 보여준다. 또한, 주의, 정서, 자아감 형성과 같은 복잡한 심리 현상까지 설명할 수 있는 인지 모델을 제공한다.",
        "question": "뇌가 예측을 통해 감각 정보를 처리한다는 주장을 시각 경험을 예로 들면 어떻게 설명할 수 있을까?",
        "bloom_level": "적용"
    },
    {
        "paragraph": "최근 연구에 따르면, 나뭇잎소리, 새소리, 물소리와 같은 자연의 소리를 듣는 것만으로도 스트레스 수치가 낮아지고 집중력이 향상될 수 있다고 한다. 실험 참가자들이 인공적인 도시 소음과 자연의 소리를 각각 들었을 때, 자연의 소리를 들은 그룹은 심박수와 코르티솔 수치가 낮아졌고, 주의 전환 속도와 기억력에서 더 높은 성과를 보였다. 연구자들은 자연의 소리가 뇌의 주의 회복 시스템을 자극해, 과도한 정보 처리에서 벗어나게 돕는다고 설명한다. 이는 단순히 조용한 환경이 주는 효과가 아니라, 자연 특유의 리듬과 패턴이 신경계에 긍정적 영향을 주기 때문으로 보인다. 이러한 연구 결과는 일상생활에서 자연 소리를 의도적으로 접하는 것이 정신 건강 증진에 실질적인 도움이 될 수 있음을 시사한다.",
        "question": "자연의 소리와 인공적인 소음 간에 존재하는 차이 중 어떤 요소가 스트레스 수치 또는 집중력 등과 관련이 있는 것일까?",
        "bloom_level": "분석"
    },
    {
        "paragraph": "텍스트 외 존재론(Ontology Outside of Text)은 해체주의 이후의 철학과 문학이론에서 등장한 개념으로, 언어 바깥의 세계와 경험을 이해하려는 시도를 말한다. 기존의 문학 이론은 주로 언어, 기호, 담론을 통해 인간의 현실을 해석했지만, 이 이론은 그것만으로는 설명되지 않는 실제 삶의 층위를 강조한다. 데리다의 해체론이 모든 의미는 언어 안에서 차이와 지연으로 구성된다고 본 반면, 텍스트 외 존재론은 언어로 포착되지 않는 감각, 몸, 침묵 같은 요소들에도 주목한다. 이 관점은 예술이나 문학에서 말로 설명되지 않는 감정이나 경험을 이해하는 데 도움을 준다. 결국 텍스트 외 존재론은 언어 중심의 사고에서 벗어나 인간 존재에 대한 보다 폭넓은 이해를 추구한다.",
        "question": "언어 중심의 해석 방법과 비교할 때 이론적으로 어떤 한계가 있을까?",
        "bloom_level": "평가"
    },
    {
        "paragraph": "기억의 장소(sites of memory)는 공동체의 역사적 경험이나 정체성이 구체적인 지리적 공간에 응축되어 저장된 장소를 의미하며, 예술은 이를 서사적으로 재구성하는 중요한 매체로 작동한다. 특히 역사적 트라우마와 같은 복잡한 주제들을 다루는 예술 작품은 과거의 사건을 현재의 감각과 윤리 속으로 불러오는 적극적인 재구성 작업을 수행한다. 예술적 재현은 공식 기록으로 남지 않은 기억의 공백을 채우고, 소외된 기억들을 복원함으로써 개인의 기억과 집단적 기억 사이의 경계를 흐리게 만든다. 이러한 작업은 관람자가 기억의 참여자이자 해석자로 전환되도록 유도한다. 이처럼 예술은 단순한 표현 수단을 넘어, 기억의 정치성과 윤리성, 사회적 기억의 구성 방식을 비판적으로 탐구하는 도구로 기능한다.",
        "question": "기억의 예술적 재현은 역사적 사실 검증과 어떤 측면에서 긴장 관계를 맺을 수 있을까?",
        "bloom_level": "창조"
    },
    {
        "paragraph": "코그니타리아트는 후기 자본주의 체제에서 인지, 정동, 창의성을 중심 자산으로 동원당하는 신(新)노동계급을 지칭하는 개념이다. 이들은 비물질적 노동의 수행 주체로서, 디지털 네트워크에 매개된 작업 환경 속에서 자기표현과 성과 창출의 무한한 자기책임성을 강요받는다. ‘자유로운 창조자’라는 표상 이면에는 플랫폼 자본주의가 조장한 노동의 유연화와 생계의 불확실성이 구조적으로 깊게 내재되어 있다. 이로써 코그니타리아트는 근대적 프롤레타리아트와 달리, 신자유주의적 자아 기술을 통해 자기 착취에 스스로를 능동적으로 동원하게 되는 존재로 전락한다. 이 개념은 노동의 본질이 물질에서 정보로 이행함에 따라, 권력과 저항의 지형 또한 근본적으로 재편되고 있음을 날카롭게 시사한다.",
        "question": "코그니타리아트가 겪는 자기 착취 문제를 줄이기 위한 새로운 노동 구조나 제도는 어떻게 설계할 수 있을까?",
        "bloom_level": "창조"
    }
]

# Few-shot examples for related question generation
RELATED_QUESTION_EXAMPLES = [
    {
        "paragraph": "‘미토포에시스(Mythopoeia)’는 단순히 기존 신화를 분석하는 데 그치지 않고, 작가가 자신만의 신화 체계를 창조하는 창작 행위를 의미한다. 이 개념은 특히 C.S. 루이스의 『나니아 연대기』에서 잘 드러나며, 그는 고유한 존재들, 종교적 상징, 윤리적 질서를 나니아라는 유기적 세계로 구성하였다.. 그의 작업은 단순한 판타지를 넘어서, 선과 악의 대립 같은 신화적 주제를 통해 인간 존재의 의미를 탐구하려는 시도였다. 이러한 미토포에시스는 고대 문명처럼 상징과 서사를 통해 세계를 설명하려는 인간의 본능과도 관련이 깊다. 현대의 문학 작품, 판타지 게임, 영화 시나리오에서도 미토포에시스는 중요한 내러티브 기법으로 활용되며, 이는 신화가 여전히 살아 있는 사유 방식임을 보여준다.",
        "user_question": "루이스는 왜 자신만의 신화를 창조하고자 했을까?",
        "suggested_question": "루이스의 신화 창작 방식에서 영감을 받아, 오늘날 우리 사회를 반영한 새로운 신화 체계를 구상하려면 어떤 세계관과 주제를 탐색해볼 수 있을까?"
    },
    {
        "paragraph": "리퀴드 모더니티(liquid modernity)는 지그문트 바우만이 제시한 개념으로, 현대 사회의 유동성과 불확실성을 설명한다. 고체적 근대가 견고한 제도와 안정된 정체성을 기반으로 했다면, 리퀴드 모더니티는 관계, 노동, 소비 방식 모두가 유동적이며 일시적인 특성을 띤다. 리퀴드 모더니티를 보이는 사회는 개인에게 유연성과 선택의 자유를 제공하지만, 동시에 지속적인 자기 재구성과 정체성의 불안을 초래한다. 이 개념은 글로벌화, 디지털화, 개인화가 지배적인 시대에서 사회적 연대와 소속감의 해체를 분석하는 데 효과적으로 활용될 수 있다. 따라서 리퀴드 모더니티는 현대인의 삶의 조건을 해석하고 사회 정책의 방향을 모색하는 데 중요한 이론적 틀을 제공한다.",
        "user_question": "고체 근대와 리퀴드 모더니티의 차이는 무엇일까?",
        "suggested_question": "고체 근대와 리퀴드 모더니티의 정체성 형성 방식 차이를 바탕으로, 디지털 플랫폼에서의 자기 표현은 어떤 새로운 윤리적 쟁점을 낳을 수 있을까?"
    },
    {
        "paragraph": "예측 처리 이론(predictive processing theory)은 인간의 뇌가 외부 자극을 받아들이기만 하는 수동적 기관이 아니라, 끊임없이 미래의 감각 정보를 예측하고 그 예측이 실제 감각 정보와 얼마나 일치하는지를 비교하면서 작동한다고 설명한다. 즉, 뇌는 예상과 다른 정보가 들어올 때 그 오류를 수정해나가는 방식으로 작동한다. 예측이 잘 맞으면 뇌의 에너지 사용은 줄어들고, 오류가 있으면 더 많은 자원이 동원되어 환경에 대한 새로운 모델을 학습하게 된다. 이 이론은 ‘정보가 입력되고 처리된다’는 고전적 인지 이론과 달리, 뇌가 능동적으로 세계를 구성한다는 관점의 전환을 보여준다. 또한, 주의, 정서, 자아감 형성과 같은 복잡한 심리 현상까지 설명할 수 있는 인지 모델을 제공한다.",
        "user_question": "뇌가 예측을 통해 감각 정보를 처리한다는 주장을 시각 경험을 예로 들면 어떻게 설명할 수 있을까?",
        "suggested_question": "시각 경험을 예로 들 때, 정서 상태가 뇌의 예측에 어떤 영향을 주는지를 알아보는 실험을 기획한다면 어떤 요소를 포함해야 할까?"
    },
    {
        "paragraph": "유전자 가위 기술, 예를 들어 CRISPR-Cas9 시스템은 특정 DNA 염기서열을 정밀하게 절단하고 편집할 수 있게 하여 생명과학 연구에 혁신을 가져왔다. 이 기술은 바이러스에 대항하는 박테리아의 면역 체계에서 유래되었으며, 연구자들은 이를 활용해 유전병 치료, 작물 개량, 생물 다양성 보존 등 다양한 분야에 적용하고 있다. 유전자 가위 기술은 기존의 유전자 조작 기술보다 훨씬 간편하고 저렴하며, 편집의 정밀도가 높아 다양한 생물학적 연구에 핵심 도구로 자리잡고 있다. 하지만 생식세포 유전자 편집과 관련된 윤리적 논쟁, 생태계에 미치는 영향 등에 대해서는 여전히 활발한 논의가 진행 중이다. CRISPR는 생명과 기술, 윤리가 얽힌 복합적 문제들을 우리에게 제기한다.",
        "user_question": "생태계의 균형을 고려하여 CRISPR 기술의 응용을 조절할 수 있는 정책 방안을 고안해본다면 어떤 요소를 고려해야 할까?",
        "suggested_question": "유전자 편집 기술이 생태계에 미치는 영향을 사전에 평가하기 위한 과학적 기준이나 윤리적 기준은 어떤 방식으로 마련될 수 있을까?"
    },
    {
        "paragraph": "자기치유 콘크리트(self-healing concrete)는 콘크리트 구조물에 균열이 발생하더라도 내부의 복원 메커니즘이 작동하여 스스로 파손 부위를 복구할 수 있도록 설계된 지능형 건축 자재다. 이 기술은 박테리아가 석회석을 생성하거나, 고분자 캡슐이 외부 자극에 반응해 복합 물질을 분출하는 등의 원리를 활용하여, 수분과 공기 침투를 막고 구조적 안정성을 연장시키는 방식으로 작동한다. 자기치유 콘크리트는 유지보수 주기를 줄이고 인프라의 전체 수명을 늘리는 데 기여하지만, 초기 제조 비용 증가, 성능의 일관성 확보 문제 등의 한계 역시 존재한다. 특히 극한 온도와 습도, 반복 진동 등 특수한 조건에서도 일관된 치유 성능을 발휘할 때, 지속가능한 도시 인프라를 구현하는 데 기여할 수 있다.",
        "user_question": "자기치유 기술을 다리나 터널 등에 적용하려면 어떤 조건을 고려해야 할까?",
        "suggested_question": "지진, 중차량 통행, 습기 변화가 잦은 지역의 다리에 적용할 자기치유 구조 시스템을 창안해본다면 어떤 방식으로 치유 메커니즘을 조정해야 할까?"
    },
    {
        "paragraph": "기억의 장소(sites of memory)는 공동체의 역사적 경험이나 정체성이 구체적인 지리적 공간에 응축되어 저장된 장소를 의미하며, 예술은 이를 서사적으로 재구성하는 중요한 매체로 작동한다. 특히 역사적 트라우마와 같은 복잡한 주제들을 다루는 예술 작품은 과거의 사건을 현재의 감각과 윤리 속으로 불러오는 적극적인 재구성 작업을 수행한다. 예술적 재현은 공식 기록으로 남지 않은 기억의 공백을 채우고, 소외된 기억들을 복원함으로써 개인의 기억과 집단적 기억 사이의 경계를 흐리게 만든다. 이러한 작업은 관람자가 기억의 참여자이자 해석자로 전환되도록 유도한다. 이처럼 예술은 단순한 표현 수단을 넘어, 기억의 정치성과 윤리성, 사회적 기억의 구성 방식을 비판적으로 탐구하는 도구로 기능한다.",
        "user_question": "기억의 예술적 재현은 역사적 사실 검증과 어떤 측면에서 긴장 관계를 맺을 수 있을까?",
        "suggested_question": "공식 역사와 충돌하는 사적인 기억을 바탕으로 다큐멘터리 연극이나 설치 작품을 구성할 때, 사실성과 허구성을 어떻게 조화시킬 수 있을까?"
    }
]

# Few-shot examples for unrelated question generation
UNRELATED_QUESTION_EXAMPLES = [
    {
        "paragraph": "‘미토포에시스(Mythopoeia)’는 단순히 기존 신화를 분석하는 데 그치지 않고, 작가가 자신만의 신화 체계를 창조하는 창작 행위를 의미한다. 이 개념은 특히 C.S. 루이스의 『나니아 연대기』에서 잘 드러나며, 그는 고유한 존재들, 종교적 상징, 윤리적 질서를 나니아라는 유기적 세계로 구성하였다.. 그의 작업은 단순한 판타지를 넘어서, 선과 악의 대립 같은 신화적 주제를 통해 인간 존재의 의미를 탐구하려는 시도였다. 이러한 미토포에시스는 고대 문명처럼 상징과 서사를 통해 세계를 설명하려는 인간의 본능과도 관련이 깊다. 현대의 문학 작품, 판타지 게임, 영화 시나리오에서도 미토포에시스는 중요한 내러티브 기법으로 활용되며, 이는 신화가 여전히 살아 있는 사유 방식임을 보여준다.",
        "user_question": "루이스는 왜 자신만의 신화를 창조하고자 했을까?",
        "suggested_question": "고대 문명이 신화를 통해 세계를 설명했던 방식은 현대 사회의 어떤 문제들을 새로운 서사로 다시 말하는 데 어떻게 활용될 수 있을까?"
    },
    {
        "paragraph": "리퀴드 모더니티(liquid modernity)는 지그문트 바우만이 제시한 개념으로, 현대 사회의 유동성과 불확실성을 설명한다. 고체적 근대가 견고한 제도와 안정된 정체성을 기반으로 했다면, 리퀴드 모더니티는 관계, 노동, 소비 방식 모두가 유동적이며 일시적인 특성을 띤다. 리퀴드 모더니티를 보이는 사회는 개인에게 유연성과 선택의 자유를 제공하지만, 동시에 지속적인 자기 재구성과 정체성의 불안을 초래한다. 이 개념은 글로벌화, 디지털화, 개인화가 지배적인 시대에서 사회적 연대와 소속감의 해체를 분석하는 데 효과적으로 활용될 수 있다. 따라서 리퀴드 모더니티는 현대인의 삶의 조건을 해석하고 사회 정책의 방향을 모색하는 데 중요한 이론적 틀을 제공한다.",
        "user_question": "고체 근대와 리퀴드 모더니티의 차이는 무엇일까?",
        "suggested_question": "리퀴드 모더니티가 지배하는 사회에서 ‘소속감’의 개념을 새롭게 정의하고 이를 측정하는 방법을 고안한다면 어떤 기준이 필요할까?"
    },
    {
        "paragraph": "예측 처리 이론(predictive processing theory)은 인간의 뇌가 외부 자극을 받아들이기만 하는 수동적 기관이 아니라, 끊임없이 미래의 감각 정보를 예측하고 그 예측이 실제 감각 정보와 얼마나 일치하는지를 비교하면서 작동한다고 설명한다. 즉, 뇌는 예상과 다른 정보가 들어올 때 그 오류를 수정해나가는 방식으로 작동한다. 예측이 잘 맞으면 뇌의 에너지 사용은 줄어들고, 오류가 있으면 더 많은 자원이 동원되어 환경에 대한 새로운 모델을 학습하게 된다. 이 이론은 ‘정보가 입력되고 처리된다’는 고전적 인지 이론과 달리, 뇌가 능동적으로 세계를 구성한다는 관점의 전환을 보여준다. 또한, 주의, 정서, 자아감 형성과 같은 복잡한 심리 현상까지 설명할 수 있는 인지 모델을 제공한다.",
        "user_question": "뇌가 예측을 통해 감각 정보를 처리한다는 주장을 시각 경험을 예로 들면 어떻게 설명할 수 있을까?",
        "suggested_question": "뇌가 반복적으로 예측에 실패할 때, 외부 세계에 대한 인식은 어떻게 변할 수 있는지 시뮬레이션 실험을 설계해볼 수 있을까?"
    },
    {
        "paragraph": "유전자 가위 기술, 예를 들어 CRISPR-Cas9 시스템은 특정 DNA 염기서열을 정밀하게 절단하고 편집할 수 있게 하여 생명과학 연구에 혁신을 가져왔다. 이 기술은 바이러스에 대항하는 박테리아의 면역 체계에서 유래되었으며, 연구자들은 이를 활용해 유전병 치료, 작물 개량, 생물 다양성 보존 등 다양한 분야에 적용하고 있다. 유전자 가위 기술은 기존의 유전자 조작 기술보다 훨씬 간편하고 저렴하며, 편집의 정밀도가 높아 다양한 생물학적 연구에 핵심 도구로 자리잡고 있다. 하지만 생식세포 유전자 편집과 관련된 윤리적 논쟁, 생태계에 미치는 영향 등에 대해서는 여전히 활발한 논의가 진행 중이다. CRISPR는 생명과 기술, 윤리가 얽힌 복합적 문제들을 우리에게 제기한다.",
        "user_question": "생태계의 균형을 고려하여 CRISPR 기술의 응용을 조절할 수 있는 정책 방안을 고안해본다면 어떤 요소를 고려해야 할까?",
        "suggested_question": "다양한 유전자 가위 기술 중에서 특정 목적(예: 작물 개량 vs. 유전병 치료)에 더 적합한 기술은 어떤 기준으로 선택할 수 있을까?"
    },
    {
        "paragraph": "자기치유 콘크리트(self-healing concrete)는 콘크리트 구조물에 균열이 발생하더라도 내부의 복원 메커니즘이 작동하여 스스로 파손 부위를 복구할 수 있도록 설계된 지능형 건축 자재다. 이 기술은 박테리아가 석회석을 생성하거나, 고분자 캡슐이 외부 자극에 반응해 복합 물질을 분출하는 등의 원리를 활용하여, 수분과 공기 침투를 막고 구조적 안정성을 연장시키는 방식으로 작동한다. 자기치유 콘크리트는 유지보수 주기를 줄이고 인프라의 전체 수명을 늘리는 데 기여하지만, 초기 제조 비용 증가, 성능의 일관성 확보 문제 등의 한계 역시 존재한다. 특히 극한 온도와 습도, 반복 진동 등 특수한 조건에서도 일관된 치유 성능을 발휘할 때, 지속가능한 도시 인프라를 구현하는 데 기여할 수 있다.",
        "user_question": "자기치유 기술을 다리나 터널 등에 적용하려면 어떤 조건을 고려해야 할까?",
        "suggested_question": "수분과 공기의 침투를 최소화할 수 있는 새로운 형태의 자기치유 콘크리트를 설계한다면, 어떤 복합 재료와 메커니즘을 조합할 수 있을까?"
    },
    {
        "paragraph": "기억의 장소(sites of memory)는 공동체의 역사적 경험이나 정체성이 구체적인 지리적 공간에 응축되어 저장된 장소를 의미하며, 예술은 이를 서사적으로 재구성하는 중요한 매체로 작동한다. 특히 역사적 트라우마와 같은 복잡한 주제들을 다루는 예술 작품은 과거의 사건을 현재의 감각과 윤리 속으로 불러오는 적극적인 재구성 작업을 수행한다. 예술적 재현은 공식 기록으로 남지 않은 기억의 공백을 채우고, 소외된 기억들을 복원함으로써 개인의 기억과 집단적 기억 사이의 경계를 흐리게 만든다. 이러한 작업은 관람자가 기억의 참여자이자 해석자로 전환되도록 유도한다. 이처럼 예술은 단순한 표현 수단을 넘어, 기억의 정치성과 윤리성, 사회적 기억의 구성 방식을 비판적으로 탐구하는 도구로 기능한다.",
        "user_question": "기억의 예술적 재현은 역사적 사실 검증과 어떤 측면에서 긴장 관계를 맺을 수 있을까?",
        "suggested_question": "‘기억의 장소’ 개념을 물리적 장소가 아닌 ‘심리적 상태’로 해석한다면, 예술 작품은 어떤 식으로 트라우마를 재구성할 수 있을까?"
    }
]

# Set this to False to disable parallel port for initial testing
USE_PARALLEL_PORT = False  # Change to True when you're ready to test with actual hardware

if USE_PARALLEL_PORT:
    try:
        from psychopy import parallel
        # Check if ParallelPort exists and is callable
        if hasattr(parallel, 'ParallelPort') and callable(parallel.ParallelPort):
            port = parallel.ParallelPort(address=0x378)
            PARALLEL_PORT_AVAILABLE = True
            print("Parallel port initialized successfully")
        else:
            print("ParallelPort class not available in psychopy.parallel")
            PARALLEL_PORT_AVAILABLE = False
            port = None
    except Exception as e:
        print(f"Parallel port initialization failed: {e}")
        PARALLEL_PORT_AVAILABLE = False
        port = None
else:
    PARALLEL_PORT_AVAILABLE = False
    port = None
    print("Parallel port disabled for testing")

# Event marker values (adjust as needed)
MARKERS = {
    "paragraph_start": 1,
    "paragraph_end": 2,
    "novelty_survey_start": 3,
    "novelty_survey_end": 4,
    "question_input_start": 5,
    "question_input_end": 6,
    "feedback_start": 7,
    "feedback_end": 8,
    "survey_start": 9,
    "survey_end": 10,
    "edit_start": 11,
    "edit_end": 12
}

# Function to send marker through parallel port
def send_marker(marker_type):
    if PARALLEL_PORT_AVAILABLE and port is not None:
        try:
            port.setData(MARKERS[marker_type])
            time.sleep(0.05)  # Brief pulse
            port.setData(0)  # Reset
        except Exception as e:
            st.error(f"Error sending marker: {e}")
    else:
        # When parallel port is disabled, just log the marker event
        if USE_PARALLEL_PORT:
            # If we tried to use parallel port but it failed
            print(f"Cannot send marker '{marker_type}': Parallel port not available")
        else:
            # If parallel port is intentionally disabled for testing
            print(f"[TEST MODE] Would send marker: {marker_type} (value: {MARKERS[marker_type]})")
    
    # Log the marker event regardless of parallel port availability
    log_event(f"MARKER: {marker_type}")

# Function to log events
def log_event_batched(event_description, data=None):
    """Optimized logging with batching for non-critical events"""
    if hasattr(st.session_state, 'logger'):
        st.session_state.logger.add_event(event_description, data)
    else:
        log_event(event_description, data)  # Fallback

def log_event(event_description, data=None):
    if 'event_log' not in st.session_state:
        st.session_state.event_log = []
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    log_entry = {
        "timestamp": timestamp,
        "iteration": st.session_state.iteration,
        "stage": st.session_state.stage,
        "event": event_description
    }
    
    if data:
        log_entry["data"] = data
        
    st.session_state.event_log.append(log_entry)

# Function to save logs
def save_logs():
    if 'event_log' in st.session_state and st.session_state.event_log:
        # Create directory if it doesn't exist
        if not os.path.exists("logs"):
            os.makedirs("logs")
            
        participant_id = st.session_state.get("participant_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/participant_{participant_id}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(st.session_state.event_log, f, indent=2)
        
        # Also save responses data for easy analysis
        responses_filename = f"logs/responses_{participant_id}_{timestamp}.csv"
        if 'responses' in st.session_state and st.session_state.responses:
            df = pd.DataFrame(st.session_state.responses)
            df.to_csv(responses_filename, index=False)
        
        return filename, responses_filename
    return None, None

# Function to get current CSV data for download
def get_current_csv_data():
    if 'responses' in st.session_state and st.session_state.responses:
        df = pd.DataFrame(st.session_state.responses)
        return df.to_csv(index=False)
    return ""

@st.cache_data
def create_condition_assignment(total_paragraphs=45, condition_randomization_seed=None):
    """
    Create randomized condition assignments for paragraphs with balanced topic distribution.
    Now cached to avoid recalculation.
    """
    if condition_randomization_seed is not None:
        random.seed(condition_randomization_seed)
    
    condition_mapping = {}
    conditions = ["related", "unrelated", "no_feedback"]
    
    # Process each topic (5 topics, 9 paragraphs each)
    for topic in range(5):
        topic_start = topic * 9
        topic_paragraphs = list(range(topic_start, topic_start + 9))
        
        # Shuffle paragraphs within this topic
        random.shuffle(topic_paragraphs)
        
        # Assign 3 paragraphs to each condition
        for i, para_idx in enumerate(topic_paragraphs):
            condition_idx = i // 3  # 0-2 -> condition 0, 3-5 -> condition 1, 6-8 -> condition 2
            condition_mapping[para_idx] = conditions[condition_idx]
    
    return condition_mapping

@st.cache_resource
def initialize_llm_models():
    """Cache LLM model initialization to avoid repeated API setup"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None
    
    classification_llm = ChatOpenAI(
        model="gpt-4-0613",
        temperature=0.1,
        openai_api_key=api_key,
        max_retries=2
    )
    generation_llm = ChatOpenAI(
        model="gpt-4-0613",
        temperature=0.7,
        openai_api_key=api_key,
        max_retries=2
    )
    return classification_llm, generation_llm

@st.cache_data
def get_common_words():
    """Precompute common word sets for validation"""
    return {
        '이', '그', '저', '것', '수', '있', '없', '는', '을', '를', '이', '가', '에', '의', '로', '으로', 
        '와', '과', '어떤', '어떻게', '왜', '무엇', '언제', '어디서', '어떠한', '그런', '이런', '저런',
        '하는', '되는', '있는', '없는', '같은', '다른', '새로운', '기존', '현재', '미래', '과거',
        '대한', '위한', '통해', '따라', '관련', '문제', '방법', '방식', '경우', '상황', '조건',
        '결과', '영향', '효과', '중요', '필요', '가능', '연구', '분석', '탐구', '제안', '개발',
        '창조', '혁신', '아이디어', '해결', '답', '질문', '생각', '고려', '검토', '평가'
    }

@lru_cache(maxsize=1000)
def get_content_words(text):
    """Cache content word extraction"""
    common_words = get_common_words()
    words = set(text.replace('?', '').replace('.', '').replace(',', '').lower().split())
    return words - common_words

def create_bloom_classification_chain(llm):
    """Create a chain for classifying questions according to Bloom's taxonomy with structured output."""
    
    # Create output parser
    parser = PydanticOutputParser(pydantic_object=BloomClassification)
    
    # Create example selector for few-shot prompting
    example_selector = LengthBasedExampleSelector(
        examples=BLOOM_CLASSIFICATION_EXAMPLES,
        example_prompt=PromptTemplate(
            input_variables=["paragraph", "question", "bloom_level"],
            template="Paragraph: {paragraph}\nQuestion: {question}\nBloom Level: {bloom_level}"
        ),
        max_length=2000,
    )
    
    # Create few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate(
            input_variables=["paragraph", "question", "bloom_level"],
            template="Paragraph: {paragraph}\nQuestion: {question}\nBloom Level: {bloom_level}"
        ),
        prefix="""다음은 Bloom's Taxonomy를 사용하여 질문을 분류하는 예시들입니다:

Bloom's Taxonomy 6단계:
1. 기억: 텍스트 내용을 기억하기 위한 질문 
2. 이해: 텍스트 내용을 바탕으로 대답할 수 있는 질문; 사실이나 이해, 정의에 대한 질문
3. 적용: 텍스트에 대해 추가적인 내용을 질문하거나 (방법, 선행문헌 등) 비슷하지만 다른 상황에 적용하는 질문
4. 분석: 텍스트의 내용 요소 간 연결 관계, 인과 관계 등을 묻는 질문, 텍스트 및 저자들의 의도를 묻는 질문
5. 평가: (배경지식을 활용하여) 텍스트에 대한 판단 및 비평을 제안하는 질문
6. 창조: 창의적인 연구 가설 또는 연구를 할 수 있는 새로운 방향을 제안하는 질문

예시들:""",
        suffix="""이제 다음 질문을 분류해주세요.

Paragraph: {paragraph}
Question: {question}

{format_instructions}

분류 결과:""",
        input_variables=["paragraph", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return LLMChain(
        llm=llm,
        prompt=few_shot_prompt,
        output_key="bloom_classification",
        output_parser=parser
    )

def create_related_question_generation_chain(llm):
    """Create a chain for generating related questions using structured output."""
    
    # Create output parser
    parser = PydanticOutputParser(pydantic_object=QuestionSuggestion)
    
    example_selector = LengthBasedExampleSelector(
        examples=RELATED_QUESTION_EXAMPLES,
        example_prompt=PromptTemplate(
            input_variables=["paragraph", "user_question", "suggested_question"],
            template="Paragraph: {paragraph}\nUser Question: {user_question}\nSuggested Question: {suggested_question}"
        ),
        max_length=1500,
    )
    
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate(
            input_variables=["paragraph", "user_question", "suggested_question"],
            template="Paragraph: {paragraph}\nUser Question: {user_question}\nSuggested Question: {suggested_question}"
        ),
        prefix="""다음은 사용자의 질문 요소를 활용해 '창조' 수준의 질문을 제안하는 예시들입니다:

중요 지침: 
- 반드시 사용자의 원래 질문에서 핵심 단어나 개념을 포함해야 합니다
- 사용자 질문을 확장하고 발전시키는 방향으로 작성하세요
- 완전히 새로운 주제로 바꾸지 마세요
- 사용자의 관심사와 접근법을 더 깊이 탐구하세요

예시들:""",
        suffix="""이제 다음 조건을 반드시 *모두* 따라 새로운 질문을 하나만 제안해주세요:

핵심 원칙: 학습자의 기존 질문을 발전시키고 확장하는 방향으로 질문을 구성하세요.

조건:
1. 학습자가 기존에 제시한 질문(question)의 핵심 키워드와 주제를 반드시 포함해야 함
2. 기존 질문에서 제기한 관점이나 접근법을 더 깊이 있게 탐구하는 방향
3. 기존 질문 + paragraph의 새로운 내용을 결합하여 확장된 질문 구성
4. Bloom's taxonomy에서 '창조' 수준의 질문 (새롭고 창의적인 연구 문제를 제안)
5. 대학교 학부생 수준에서 이해 가능해야 함
6. 질문은 한국어로 한 문장이어야 함 (글자수 65-75자)
7. 물음표로 끝나야 함

금지사항:
- 기존 질문과 완전히 다른 주제로 바꾸는 것
- 기존 질문의 핵심 개념을 무시하는 것
- 기존 질문보다 단순한 수준의 질문

중요: 학습자의 원래 질문 "{question}"의 핵심 요소를 반드시 포함하고 발전시켜야 합니다.

Paragraph: {paragraph}
User Question: {question}

{format_instructions}

새로운 질문 (기존 질문을 발전시킨 버전):""",
        input_variables=["paragraph", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return LLMChain(
        llm=llm,
        prompt=few_shot_prompt,
        output_key="question_suggestion",
        output_parser=parser
    )

def create_unrelated_question_generation_chain(llm):
    """Create a chain for generating unrelated questions using structured output."""
    
    # Create output parser
    parser = PydanticOutputParser(pydantic_object=QuestionSuggestion)
    
    example_selector = LengthBasedExampleSelector(
        examples=UNRELATED_QUESTION_EXAMPLES,
        example_prompt=PromptTemplate(
            input_variables=["paragraph", "user_question", "suggested_question"],
            template="Paragraph: {paragraph}\nUser Question: {user_question}\nSuggested Question: {suggested_question}"
        ),
        max_length=1500,
    )
    
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate(
            input_variables=["paragraph", "user_question", "suggested_question"],
            template="Paragraph: {paragraph}\nUser Question: {user_question}\nSuggested Question: {suggested_question}"
        ),
        prefix="""다음은 사용자의 질문과 무관하게 paragraph만을 기반으로 '창조' 수준의 질문을 제안하는 예시들입니다:

중요 지침:
- 사용자의 원래 질문에서 사용된 단어나 개념을 절대 사용하지 마세요
- 사용자 질문과는 완전히 다른 각도에서 접근하세요
- 사용자 질문을 발전시키거나 확장하지 마세요
- paragraph의 다른 측면이나 요소에 집중하세요

예시들:""",
        suffix="""이제 다음 조건을 반드시 *모두* 따라 새로운 질문을 하나만 제안해주세요:

핵심 원칙: 제시된 Paragraph 내에서, 학습자의 기존 질문과는 완전히 다른 관점을 탐구하는 질문을 구성하세요.

조건:
1. 반드시 제시된 Paragraph의 내용과 직접 관련된 질문이어야 함
2. 학습자가 기존에 제시한 질문(question)의 키워드, 주제, 접근법을 일절 사용하지 말 것
3. Paragraph에서 기존 질문이 다루지 않은 완전히 다른 측면이나 요소를 선택
4. 같은 텍스트 내의 다른 개념, 인물, 시대, 방법론, 분야 등에 집중
5. Bloom's taxonomy에서 '창조' 수준의 질문 (새롭고 창의적인 연구 문제를 제안)
6. 대학교 학부생 수준에서 이해 가능해야 함
7. 질문은 한국어로 한 문장이어야 함 (글자수 65-75자)
8. 물음표로 끝나야 함

금지사항:
- Paragraph 범위를 벗어나 완전히 다른 주제로 가는 것
- 기존 질문에서 언급된 개념이나 단어 재사용
- Paragraph에 없는 내용을 추가하는 것

전략: Paragraph를 다시 읽고, 사용자가 주목하지 않은 다른 요소(인물, 시대적 배경, 다른 개념, 응용 분야, 사회적 함의 등)를 찾아 질문하세요.

중요: 학습자의 원래 질문 "{question}"과는 완전히 무관하지만, Paragraph 내용에는 반드시 기반해야 합니다.

Paragraph: {paragraph}
User Question: {question}

{format_instructions}

새로운 질문 (기존 질문과 무관한 새로운 관점):""",
        input_variables=["paragraph", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return LLMChain(
        llm=llm,
        prompt=few_shot_prompt,
        output_key="question_suggestion",
        output_parser=parser
    )

def get_fallback_question(feedback_type, original_question):
    """Generate appropriate fallback questions when validation fails."""
    if feedback_type == "related":
        # Extract a key concept from original question for fallback
        words = original_question.replace('?', '').split()
        content_words = [w for w in words if len(w) > 2 and w not in ['어떤', '어떻게', '무엇', '왜']]
        if content_words:
            key_concept = content_words[0]
            return f"{key_concept}을 바탕으로 새로운 연구 방향을 제안해볼 수 있을까?"
        else:
            return "이 개념을 바탕으로 새로운 연구 방향을 제안해볼 수 있을까?"
    else:  # unrelated
        # Paragraph-grounded fallback questions that stay within the text scope
        fallback_questions = [
            "이 주제의 다른 측면을 새롭게 탐구할 수 있는 방법은 무엇일까?",
            "텍스트에서 다루지 않은 관련 요소를 발전시킬 수 있을까?",
            "이 개념을 다른 방향으로 확장해볼 수 있는 방안은 무엇일까?",
            "텍스트 내 다른 관점에서 새로운 접근법을 제안할 수 있을까?"
        ]
        import random
        return random.choice(fallback_questions)

def get_bloom_classification_with_fallback(llm, paragraph, question, max_retries=2):
    """Get Bloom classification with optimized retry logic"""
    classification_chain = create_bloom_classification_chain(llm)
    
    for attempt in range(max_retries):
        try:
            result = classification_chain.run({"paragraph": paragraph, "question": question})
            
            # Extract bloom level
            if hasattr(result, 'bloom_level'):
                return result.bloom_level
            elif isinstance(result, dict) and 'bloom_level' in result:
                return result['bloom_level']
            else:
                bloom_level = str(result).strip()
                if bloom_level:
                    return bloom_level
                    
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Classification failed after {max_retries} attempts: {e}")
            continue
    
    return "기억"  # Default fallback

def generate_question_without_validation(llm, paragraph, question, feedback_type, max_retries=3):
    """Generate question without validation but with metrics collection"""
    
    # Create appropriate chain
    if feedback_type == "related":
        chain = create_related_question_generation_chain(llm)
    else:
        chain = create_unrelated_question_generation_chain(llm)
    
    for attempt in range(max_retries):
        try:
            result = chain.run({"paragraph": paragraph, "question": question})
            
            # Extract question
            if hasattr(result, 'suggested_question'):
                suggested_question = result.suggested_question
            elif isinstance(result, dict) and 'suggested_question' in result:
                suggested_question = result['suggested_question']
            else:
                suggested_question = str(result).strip()
            
            # Basic format validation (just ensure it's not empty and has question mark)
            if suggested_question and len(suggested_question.strip()) > 0:
                if not suggested_question.endswith('?'):
                    suggested_question += '?'
                return suggested_question
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            continue
    
    # Fallback if all attempts failed
    return get_fallback_question(feedback_type, question)

def calculate_question_metrics(original_question, suggested_question, paragraph):
    """Calculate relatedness and other metrics for storage without validation"""
    
    # Calculate relatedness score
    original_content = get_content_words(original_question)
    suggested_content = get_content_words(suggested_question)
    
    if not original_content:
        relatedness_score = 0.0
    else:
        # Calculate overlap ratios
        overlap = len(original_content & suggested_content)
        overlap_ratio = overlap / len(original_content)
        
        # Concept overlap check
        concept_overlap = sum(1 for word in original_content if len(word) > 2 and word in suggested_question.lower())
        concept_ratio = concept_overlap / len(original_content)
        
        relatedness_score = max(overlap_ratio, concept_ratio)
    
    # Calculate paragraph relevance
    paragraph_content = get_content_words(paragraph)
    question_content = get_content_words(suggested_question)
    
    if not question_content:
        paragraph_relevance = 0.0
    else:
        overlap = len(paragraph_content & question_content)
        paragraph_relevance = overlap / len(question_content)
    
    # Calculate length
    question_length = len(suggested_question)
    
    # Calculate word count
    question_word_count = len(suggested_question.split())
    
    return {
        'relatedness_score': round(relatedness_score, 3),
        'paragraph_relevance': round(paragraph_relevance, 3),
        'question_length': question_length,
        'question_word_count': question_word_count,
        'ends_with_question_mark': suggested_question.endswith('?'),
        'is_empty': len(suggested_question.strip()) == 0
    }

def handle_api_error(error, feedback_type):
    """Centralized API error handling"""
    error_msg = str(error)
    if "insufficient_quota" in error_msg or "quota" in error_msg.lower():
        if feedback_type == "no_feedback":
            return "OpenAI API quota exceeded. Using mock response: '기억' 수준의 질문을 작성하셨군요.\n다음 단계로 넘어가면 질문을 수정할 기회가 주어집니다."
        else:
            return "OpenAI API quota exceeded. Using mock response: '기억' 수준의 질문을 작성하셨군요.\n'이 내용을 바탕으로 새로운 아이디어를 제안해보세요?'와 같은 질문으로 수정하는 것은 어떨까요?"
    else:
        return f"Error generating AI feedback: {error_msg}"

# Function to get AI feedback using LangChain
def get_ai_feedback(question, paragraph, original_paragraph_index):
    """
    Optimized AI feedback generation without validation but with metrics collection
    """
    
    feedback_type = st.session_state.condition_mapping.get(original_paragraph_index, "no_feedback")
    
    try:
        # Get cached LLM models
        classification_llm, generation_llm = initialize_llm_models()
        if not classification_llm:
            return "Error: OpenAI API key not found."
        
        # STEP 1: Classification (always needed)
        bloom_level = get_bloom_classification_with_fallback(classification_llm, paragraph, question)
        
        # STEP 2: Generate suggestion only if needed
        if feedback_type == "no_feedback":
            suggested_question = None
            question_metrics = None
            final_response = f"'{bloom_level}' 수준의 질문을 작성하셨군요.\n다음 단계로 넘어가면 질문을 수정할 기회가 주어집니다. 더 창의적인 질문으로 수정하는 것은 어떨까요?"
        else:
            # Generate question without validation
            suggested_question = generate_question_without_validation(
                generation_llm, paragraph, question, feedback_type
            )
            
            # Calculate metrics for storage (but don't use for validation)
            question_metrics = calculate_question_metrics(question, suggested_question, paragraph)
            
            final_response = f"'{bloom_level}' 수준의 질문을 작성하셨군요.\n'{suggested_question}'와 같은 질문으로 수정하는 것은 어떨까요?"
        
        # Store metrics in session state for later CSV inclusion
        if question_metrics:
            st.session_state.current_iteration_data.update({
                'suggested_question_metrics': question_metrics
            })
        
        # Log execution details
        log_event("AI feedback generated", {
            "bloom_level": bloom_level,
            "suggested_question": suggested_question,
            "feedback_type": feedback_type,
            "original_paragraph_index": original_paragraph_index,
            "question_metrics": question_metrics
        })
        
        return final_response
        
    except Exception as e:
        return handle_api_error(e, feedback_type)
    
def get_session_value(key, default=None):
    """Helper to safely get session state values"""
    return st.session_state.get(key, default)

def set_session_value(key, value):
    """Helper to set session state values"""
    st.session_state[key] = value

def create_widget_key(base_name, iteration=None):
    """Create consistent widget keys"""
    if iteration is None:
        iteration = get_session_value('iteration', 0)
    return f"{base_name}_{iteration}"

class EventLogger:
    def __init__(self):
        self.pending_events = []
        self.batch_size = 5  # Smaller batch for experiment context
    
    def add_event(self, event_description, data=None):
        """Add event to batch"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
        log_entry = {
            "timestamp": timestamp,
            "iteration": get_session_value('iteration', 0),
            "stage": get_session_value('stage', 'unknown'),
            "event": event_description
        }
        
        if data:
            log_entry["data"] = data
        
        self.pending_events.append(log_entry)
        
        # Flush if batch is full or for critical events
        if (len(self.pending_events) >= self.batch_size or 
            "completed" in event_description.lower() or
            "error" in event_description.lower()):
            self.flush()
    
    def flush(self):
        """Write all pending events to session state"""
        if 'event_log' not in st.session_state:
            st.session_state.event_log = []
        
        st.session_state.event_log.extend(self.pending_events)
        self.pending_events.clear()

# Initialize global logger
if 'logger' not in st.session_state:
    st.session_state.logger = EventLogger()
        
# Initialize session state variables if they don't exist
def initialize_session_state():
    if 'started' not in st.session_state:
        st.session_state.started = False
    
    if 'iteration' not in st.session_state:
        st.session_state.iteration = 0
    
    if 'stage' not in st.session_state:
        st.session_state.stage = "start"
    
    # Initialize timing variables
    if 'stage_timers' not in st.session_state:
        st.session_state.stage_timers = {}
    
    if 'paragraphs' not in st.session_state:
        # Get paragraphs from config (now 45 paragraphs)
        original_paragraphs = get_paragraphs(45)
        
        # Create paragraph order
        # Paragraph randomization settings
        RANDOMIZE_PARAGRAPHS = True  # Set to False for sequential order
        RANDOM_SEED = None  # Set to an integer for reproducible randomization, or None for different each time
        
        if RANDOMIZE_PARAGRAPHS:
            # Set seed if specified (for reproducible randomization)
            if RANDOM_SEED is not None:
                random.seed(RANDOM_SEED)
            
            # Create a randomized order of paragraph indices
            paragraph_indices = list(range(len(original_paragraphs)))
            random.shuffle(paragraph_indices)
            
            # Store both the original paragraphs and the randomized indices
            st.session_state.original_paragraphs = original_paragraphs
            st.session_state.paragraph_indices = paragraph_indices
            
            # Create the randomized paragraph list
            st.session_state.paragraphs = [original_paragraphs[i] for i in paragraph_indices]
            
            # Store the mapping for logging and analysis
            st.session_state.paragraph_mapping = {i: paragraph_indices[i] for i in range(len(paragraph_indices))}
            
            print(f"Paragraphs randomized. Seed: {RANDOM_SEED if RANDOM_SEED is not None else 'random'}")
        else:
            # Use sequential paragraphs
            st.session_state.paragraphs = original_paragraphs
            st.session_state.paragraph_mapping = {i: i for i in range(len(original_paragraphs))}
            
            print("Paragraphs in sequential order.")
    
    if 'responses' not in st.session_state:
        st.session_state.responses = []
    
    # Initialize current iteration data storage
    if 'current_iteration_data' not in st.session_state:
        st.session_state.current_iteration_data = {}

# Function to start stage timer
def start_stage_timer(stage_name):
    st.session_state.stage_timers[f"{stage_name}_start"] = time.time()

# Function to end stage timer and calculate duration
def end_stage_timer(stage_name):
    start_key = f"{stage_name}_start"
    duration_key = f"{stage_name}_duration"
    
    if start_key in st.session_state.stage_timers:
        duration = time.time() - st.session_state.stage_timers[start_key]
        st.session_state.stage_timers[duration_key] = duration
        return duration
    return 0

# Function to advance to the next stage
def next_stage(next_stage_name):
    # End timer for current stage
    end_stage_timer(st.session_state.stage)
    
    # Log the stage transition
    log_event(f"Stage transition: {st.session_state.stage} -> {next_stage_name}")
    st.session_state.stage = next_stage_name

    time.sleep(0.1)
    
    # Start timer for next stage
    start_stage_timer(next_stage_name)
    
    st.rerun()

# Function to handle the start of a new iteration
def start_iteration():
    if st.session_state.iteration >= 45:
        st.session_state.stage = "completed"
        log_event("Experiment completed")
    else:
        st.session_state.stage = "show_paragraph"
        start_stage_timer("show_paragraph")
        send_marker("paragraph_start")
        log_event("Iteration started", {"iteration_number": st.session_state.iteration})

# Function to handle the completion of viewing paragraph
def paragraph_viewed():
    send_marker("paragraph_end")
    send_marker("novelty_survey_start")
    next_stage("novelty_survey")

# Function to handle novelty survey submission
def submit_novelty_survey():
    send_marker("novelty_survey_end")
    
    # Get novelty rating and comments
    novelty_rating = st.session_state.get('novelty_rating')
    paragraph_comments = st.session_state.get('paragraph_comments', '')
    difficulty_rating = st.session_state.get('difficulty_rating')
    difficulty_comments = st.session_state.get('difficulty_comments', '')
    
    if novelty_rating is None:
        st.error("Please rate how novel the paragraph is before proceeding.")
        return
    
    if difficulty_rating is None:
        st.error("Please rate how difficult the paragraph is before proceeding.")
        return
    
    # Store in current iteration data
    st.session_state.current_iteration_data['novelty_rating'] = novelty_rating
    st.session_state.current_iteration_data['paragraph_comments'] = paragraph_comments
    st.session_state.current_iteration_data['difficulty_rating'] = difficulty_rating
    st.session_state.current_iteration_data['difficulty_comments'] = difficulty_comments
    
    # Log novelty and difficulty survey data
    log_event_batched("Novelty and difficulty survey submitted", {
        "novelty_rating": novelty_rating,
        "paragraph_comments": paragraph_comments,
        "difficulty_rating": difficulty_rating,
        "difficulty_comments": difficulty_comments
    })
    
    # Move to question input stage
    send_marker("question_input_start")
    next_stage("ask_question")

# Function to log textarea focus events
# Function to log textarea focus events
def log_textarea_focus(textarea_type):
    """Log when a textarea is focused/clicked"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    if textarea_type == "edit_question":
        # Store the focus time for edit question specifically
        st.session_state.edit_textarea_focus_time = time.time()
        send_marker("edit_textarea_focus")
    elif textarea_type == "question_input":
        # Store the focus time for question input specifically
        st.session_state.question_input_focus_time = time.time()
    
    log_event(f"Textarea focus: {textarea_type}", {
        "focus_timestamp": timestamp,
        "focus_time": time.time()
    })

# Function to handle question submission
def submit_question():
    # Get the question from session state (it should exist now)
    question = st.session_state.get('user_question', '')
    question_comments = st.session_state.get('question_comments', '')
    
    # Enhanced validation
    if not question.strip():
        st.error("질문을 입력해주세요.")
        return
    
    # Check if the question is only a question mark (with optional whitespace)
    if question.strip() == '?':
        st.error("질문을 입력해주세요.")
        return
    
    # Calculate question input interaction time (from first focus to submission)
    question_input_interaction_time = None
    if hasattr(st.session_state, 'question_input_focus_time'):
        question_input_interaction_time = time.time() - st.session_state.question_input_focus_time
    
    # Store the question in current iteration data for persistence
    st.session_state.current_iteration_data['user_question'] = question
    st.session_state.current_iteration_data['question_comments'] = question_comments
    st.session_state.current_iteration_data['question_input_interaction_time'] = question_input_interaction_time
    
    send_marker("question_input_end")
    
    # Get paragraph information including original index
    current_paragraph = st.session_state.paragraphs[st.session_state.iteration]
    original_paragraph_index = st.session_state.paragraph_mapping.get(st.session_state.iteration, st.session_state.iteration)
    
    # Log the submitted question with paragraph information
    log_event_batched("Question submitted", {
        "question": question,
        "question_comments": question_comments,
        "question_input_interaction_time": question_input_interaction_time,
        "paragraph_index": st.session_state.iteration,
        "original_paragraph_index": original_paragraph_index,
        "paragraph": current_paragraph
    })
    
    # Get AI feedback with conditional logic
    send_marker("feedback_start")
    feedback = get_ai_feedback(
        question, 
        current_paragraph, 
        original_paragraph_index
    )
    send_marker("feedback_end")
    
    # Store the feedback
    st.session_state.current_iteration_data['feedback'] = feedback
    
    # Log the feedback with experimental condition info
    feedback_type = st.session_state.condition_mapping.get(original_paragraph_index, "no_feedback")
    log_event("AI feedback generated", {
        "feedback": feedback,
        "feedback_type": feedback_type,
        "original_paragraph_index": original_paragraph_index
    })
    
    next_stage("show_feedback")

# Function to handle survey submission
def submit_survey():
    send_marker("survey_end")
    
    # Get the current feedback condition
    original_paragraph_index = st.session_state.paragraph_mapping.get(st.session_state.iteration, st.session_state.iteration)
    feedback_type = st.session_state.condition_mapping.get(original_paragraph_index, "no_feedback")
    
    # Get survey responses
    curiosity = st.session_state.get('curiosity')
    relatedness = st.session_state.get('relatedness')
    accept_feedback = st.session_state.get('accept_feedback')
    feedback_comments = st.session_state.get('feedback_comments', '')
    survey_comments = st.session_state.get('survey_comments', '')
    
    # Validate that required fields are filled based on condition
    if curiosity is None:
        st.error("Please rate your curiosity level before proceeding.")
        return
    if accept_feedback is None:
        st.error("Please indicate whether you accept the feedback before proceeding.")
        return
    
    # Only validate relatedness and accept_feedback for conditions other than "no_feedback"
    if feedback_type != "no_feedback":
        if relatedness is None:
            st.error("Please rate the relatedness before proceeding.")
            return
    
    # Store in current iteration data
    st.session_state.current_iteration_data['curiosity'] = curiosity
    st.session_state.current_iteration_data['relatedness'] = relatedness
    st.session_state.current_iteration_data['accept_feedback'] = accept_feedback
    st.session_state.current_iteration_data['feedback_comments'] = feedback_comments
    st.session_state.current_iteration_data['survey_comments'] = survey_comments
    
    # Log the survey responses with paragraph information
    survey_data = {
        "curiosity": curiosity,
        "relatedness": relatedness,
        "accept_feedback": accept_feedback,
        "feedback_comments": feedback_comments,
        "survey_comments": survey_comments,
        "paragraph_index": st.session_state.iteration,
        "original_paragraph_index": original_paragraph_index,
        "feedback_type": feedback_type  # Add feedback type to the log
    }
    log_event_batched("Survey submitted", survey_data)
    
    # Go to edit question stage
    send_marker("edit_start")
    next_stage("edit_question")

# Function to handle edited question submission
def submit_edited_question():
    send_marker("edit_end")
    
    # Get the edited question and comments
    edited_question = st.session_state.get('edited_question', '')
    edit_comments = st.session_state.get('edit_comments', '')
    
    if not edited_question.strip():
        st.error("Please enter a question before proceeding.")
        return
    
    # Store in current iteration data
    st.session_state.current_iteration_data['edited_question'] = edited_question
    st.session_state.current_iteration_data['edit_comments'] = edit_comments
    
    # Log the edited question
    log_event("Edited question submitted", {
        "edited_question": edited_question,
        "edit_comments": edit_comments
    })
    
    # Get the current paragraph and its original index (if randomized)
    current_paragraph = st.session_state.paragraphs[st.session_state.iteration]
    original_paragraph_index = st.session_state.paragraph_mapping.get(st.session_state.iteration, st.session_state.iteration)
    
    # **FIX: End the current stage timer BEFORE calculating durations**
    end_stage_timer(st.session_state.stage)  # This should be "edit_question"
    
    # Calculate stage durations
    stage_durations = {}
    for stage in ['show_paragraph', 'novelty_survey', 'ask_question', 'show_feedback', 'survey', 'edit_question']:
        duration_key = f"{stage}_duration"
        if duration_key in st.session_state.stage_timers:
            stage_durations[f"{stage}_time_seconds"] = st.session_state.stage_timers[duration_key]
    
    # Calculate edit textarea interaction time (from first focus to submission)
    edit_textarea_interaction_time = None
    if hasattr(st.session_state, 'edit_textarea_focus_time'):
        edit_textarea_interaction_time = time.time() - st.session_state.edit_textarea_focus_time
    
    # Get metrics if they exist
    metrics = st.session_state.current_iteration_data.get('suggested_question_metrics', {})
    
    # Store all the data for this iteration
    iteration_data = {
        "iteration": st.session_state.iteration,
        "paragraph": current_paragraph,
        "paragraph_index": st.session_state.iteration,  # Current presentation order
        "original_paragraph_index": original_paragraph_index,  # Original index before randomization
        "feedback_type": st.session_state.condition_mapping.get(original_paragraph_index, "no_feedback"),
        "novelty_rating": st.session_state.current_iteration_data.get('novelty_rating'),
        "paragraph_comments": st.session_state.current_iteration_data.get('paragraph_comments', ''),
        "difficulty_rating": st.session_state.current_iteration_data.get('difficulty_rating'),
        "difficulty_comments": st.session_state.current_iteration_data.get('difficulty_comments', ''),
        "original_question": st.session_state.current_iteration_data.get('user_question', ''),
        "question_comments": st.session_state.current_iteration_data.get('question_comments', ''),
        "question_input_interaction_time_seconds": st.session_state.current_iteration_data.get('question_input_interaction_time'),
        "feedback": st.session_state.current_iteration_data.get('feedback', ''),
        "feedback_comments": st.session_state.current_iteration_data.get('feedback_comments', ''),
        "curiosity": st.session_state.current_iteration_data.get('curiosity'),
        "relatedness": st.session_state.current_iteration_data.get('relatedness'),
        "accept_feedback": st.session_state.current_iteration_data.get('accept_feedback'),
        "survey_comments": st.session_state.current_iteration_data.get('survey_comments', ''),
        "edited_question": edited_question,
        "edit_comments": edit_comments,
        "edit_textarea_interaction_time_seconds": edit_textarea_interaction_time,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # Add question metrics to CSV
        "suggested_question_relatedness_score": metrics.get('relatedness_score'),
        "suggested_question_paragraph_relevance": metrics.get('paragraph_relevance'),
        "suggested_question_length": metrics.get('question_length'),
        "suggested_question_word_count": metrics.get('question_word_count'),
        "suggested_question_ends_with_question_mark": metrics.get('ends_with_question_mark'),
        "suggested_question_is_empty": metrics.get('is_empty'),
        **stage_durations  # Add all stage durations
    }
    st.session_state.responses.append(iteration_data)
    
    # Clear current iteration data and stage timers for next iteration
    st.session_state.current_iteration_data = {}
    st.session_state.stage_timers = {}

    # Clear edit textarea focus time
    if hasattr(st.session_state, 'edit_textarea_focus_time'):
        delattr(st.session_state, 'edit_textarea_focus_time')
    
    # Clear question input focus time
    if hasattr(st.session_state, 'question_input_focus_time'):
        delattr(st.session_state, 'question_input_focus_time')
    
    # Reset widget keys by removing them from session state
    widget_keys_to_reset = [
        'user_question', 'edited_question',
        'paragraph_comments', 'question_comments', 'feedback_comments', 
        'survey_comments', 'edit_comments', 'novelty_rating', 'difficulty_rating',
        'difficulty_comments', 'curiosity', 'relatedness', 'accept_feedback'
    ]
    
    for key in widget_keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    # Move to next iteration
    if hasattr(st.session_state, 'logger'):
        st.session_state.logger.flush()
    st.session_state.iteration += 1
    start_iteration()

# Main app
def main():
    st.title("[파일럿 연구] 생성형 AI의 피드백 유형이 질문 수정에 미치는 영향")
    
    # Initialize session state
    initialize_session_state()
    
    # Add download button in sidebar for current progress
    with st.sidebar:
        st.header("실험 진행 상황")
        if st.session_state.get('started', False):
            st.write(f"현재 반복: {st.session_state.iteration + 1}/45")
            st.write(f"현재 단계: {st.session_state.stage}")
            
            # Download current progress
            if st.session_state.responses:
                csv_data = get_current_csv_data()
                if csv_data:
                    st.download_button(
                        label="현재까지 결과 다운로드 (CSV)",
                        data=csv_data,
                        file_name=f"partial_results_{st.session_state.get('participant_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="sidebar_download"
                    )
    
    # Check if the experiment has started
    if not st.session_state.started:
        st.write("Welcome to the experiment!")
        # Researcher settings
        with st.expander("Researcher Settings", expanded=False):
            st.warning("These settings are for researchers only. Configure them before starting the experiment.")
            
            # Randomization settings
            st.subheader("Paragraph Presentation")
            randomize = st.checkbox("Randomize paragraph order", value=True)
            
            seed_option = st.radio("Randomization seed", ["Random each time", "Fixed seed"], index=0)
            fixed_seed = None
            if seed_option == "Fixed seed":
                fixed_seed = st.number_input("Enter seed value (integer)", value=42, step=1)
            
            # Condition randomization settings
            st.subheader("Condition Assignment")
            condition_seed_option = st.radio("Condition randomization seed", ["Random each time", "Fixed seed"], index=0, key="condition_seed")
            condition_fixed_seed = None
            if condition_seed_option == "Fixed seed":
                condition_fixed_seed = st.number_input("Enter condition seed value (integer)", value=123, step=1, key="condition_seed_input")
            
            # Show experimental design explanation
            st.info(f"""
            **Current Configuration:**
            - 45 paragraphs total (5 topics × 9 paragraphs each)
            - Each condition gets 3 paragraphs from each topic:
              - Related feedback: 15 paragraphs (3 from each topic)
              - Unrelated feedback: 15 paragraphs (3 from each topic)
              - No feedback: 15 paragraphs (3 from each topic)
            - Topic distribution is balanced across conditions
            - Condition assignment will be randomized within each topic
            """)
            
            # Preview condition assignment if available
            if st.session_state.get('condition_mapping'):
                with st.expander("Preview Condition Assignment", expanded=False):
                    # Group by topic and condition for display
                    topics = ["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5"]
                    for topic_idx in range(5):
                        st.write(f"**{topics[topic_idx]} (Paragraphs {topic_idx*9}-{topic_idx*9+8}):**")
                        topic_conditions = {}
                        for para_idx in range(topic_idx*9, topic_idx*9+9):
                            condition = st.session_state.condition_mapping.get(para_idx, "unknown")
                            if condition not in topic_conditions:
                                topic_conditions[condition] = []
                            topic_conditions[condition].append(para_idx)
                        
                        for condition, paragraphs in topic_conditions.items():
                            st.write(f"  - {condition}: paragraphs {paragraphs}")
                        st.write("")
            
            # Option to test parallel port
            if st.button("Test Parallel Port"):
                if PARALLEL_PORT_AVAILABLE:
                    for marker_type, value in MARKERS.items():
                        port.setData(value)
                        st.write(f"Sending marker: {marker_type} (value: {value})")
                        time.sleep(0.5)
                        port.setData(0)
                        time.sleep(0.2)
                    st.success("Parallel port test completed.")
                else:
                    st.error("Parallel port functionality not available.")
            
            # Store settings in session state for later use
            st.session_state.randomize_paragraphs = randomize
            st.session_state.random_seed = None if seed_option == "Random each time" else fixed_seed
            st.session_state.condition_randomization_seed = None if condition_seed_option == "Random each time" else condition_fixed_seed
        
        # Experiment instructions and Bloom's taxonomy explanation
        st.markdown("""
        ### 실험 안내
        
        본 실험은 주어진 텍스트를 읽고 질문을 작성한 후, AI로부터 질문에 대한 피드백을 받는 과정으로 구성되어 있습니다. 
        AI는 여러분이 작성한 질문이 Bloom의 분류체계(Bloom's Taxonomy) 중 어느 수준에 해당하는지에 대한 피드백을 제공할 것입니다.
        
        ---
                
        #### Bloom의 분류체계 6단계:
        
        **1. 기억**: 텍스트 내용을 기억하기 위한 질문\n
        **2. 이해**: 텍스트 내용을 바탕으로 대답할 수 있는 질문; 사실이나 이해, 정의에 대한 질문\n
        **3. 적용**: 텍스트에 대해 추가 내용을 질문하거나(방법, 선행문헌 등) 비슷하지만 다른 상황에 적용하는 질문\n
        **4. 분석**: 텍스트의 내용 요소 간 연결 관계, 인과 관계 등을 묻는 질문, 텍스트 및 저자들의 의도를 묻는 질문\n
        **5. 평가**: (배경지식을 활용하여) 텍스트에 대한 판단 및 비평을 제안하는 질문\n
        **6. 창조**: 창의적인 연구 가설 또는 연구를 할 수 있는 새로운 방향을 제안하는 질문\n
        
        ---
        텍스트를 읽고 질문을 작성하고, AI로부터 피드백을 받은 후, 질문을 수정하는 과정을 총 45회 반복하게 됩니다. 실험 예상 소요 시간은 1시간 15분입니다.
        실험을 시작하려면 아래에 참여자 ID를 입력하고, '실험 시작' 버튼을 눌러주세요.
        """)
        
        participant_id = st.text_input("참여자 ID를 입력해주세요(예: pilot1):")
        
        if st.button("실험 시작") and participant_id:
            st.session_state.participant_id = participant_id
            st.session_state.started = True
            
            # Create condition assignment
            st.session_state.condition_mapping = create_condition_assignment(
                total_paragraphs=45,
                condition_randomization_seed=st.session_state.get('condition_randomization_seed')
            )
            
            # Now initialize paragraphs with the chosen randomization settings
            if 'paragraphs' not in st.session_state:
                # Get paragraphs from config (now 45 paragraphs)
                original_paragraphs = get_paragraphs(45)
                
                # Create paragraph order
                # Use settings from the UI
                RANDOMIZE_PARAGRAPHS = st.session_state.randomize_paragraphs
                RANDOM_SEED = st.session_state.random_seed
                
                if RANDOMIZE_PARAGRAPHS:
                    # Set seed if specified (for reproducible randomization)
                    if RANDOM_SEED is not None:
                        random.seed(RANDOM_SEED)
                    
                    # Create a randomized order of paragraph indices
                    paragraph_indices = list(range(len(original_paragraphs)))
                    random.shuffle(paragraph_indices)
                    
                    # Store both the original paragraphs and the randomized indices
                    st.session_state.original_paragraphs = original_paragraphs
                    st.session_state.paragraph_indices = paragraph_indices
                    
                    # Create the randomized paragraph list
                    st.session_state.paragraphs = [original_paragraphs[i] for i in paragraph_indices]
                    
                    # Store the mapping for logging and analysis
                    st.session_state.paragraph_mapping = {i: paragraph_indices[i] for i in range(len(paragraph_indices))}
                    
                    # Log the randomization
                    log_event("Paragraphs randomized", {
                        "seed": RANDOM_SEED,
                        "mapping": st.session_state.paragraph_mapping
                    })
                    
                    print(f"Paragraphs randomized. Seed: {RANDOM_SEED if RANDOM_SEED is not None else 'random'}")
                else:
                    # Use sequential paragraphs
                    st.session_state.paragraphs = original_paragraphs
                    st.session_state.paragraph_mapping = {i: i for i in range(len(original_paragraphs))}
                    
                    # Log the sequential order
                    log_event("Paragraphs in sequential order")
                    
                    print("Paragraphs in sequential order.")
            
            log_event("Experiment started", {
                "participant_id": participant_id,
                "condition_mapping": st.session_state.condition_mapping,
                "randomization_settings": {
                    "randomized": st.session_state.randomize_paragraphs,
                    "seed": st.session_state.random_seed,
                    "condition_seed": st.session_state.get('condition_randomization_seed')
                }
            })
            start_iteration()
            st.rerun()
    else:
        # Display progress
        progress_bar = st.progress((st.session_state.iteration) / 45)
        st.write(f"Iteration {st.session_state.iteration + 1}/45")
        
        # Handle different stages
        if st.session_state.stage == "completed":
            st.success("Experiment completed! Thank you for your participation.")
            if hasattr(st.session_state, 'logger'):
                st.session_state.logger.flush()  # Final flush of all events
            log_files = save_logs()
            if log_files[0]:
                st.write(f"Event logs saved to: {log_files[0]}")
            if log_files[1]:
                st.write(f"Response data saved to: {log_files[1]}")
            
            # Display a summary of the responses if needed
            if st.checkbox("Show response summary"):
                df = pd.DataFrame(st.session_state.responses)
                st.write(df)
                
            # Option to download CSV
            if 'responses' in st.session_state and st.session_state.responses:
                df = pd.DataFrame(st.session_state.responses)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name=f"results_{st.session_state.participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        elif st.session_state.stage == "show_paragraph":
            # Display the paragraph
            st.subheader("다음 텍스트를 읽어주세요:")
            st.write(st.session_state.paragraphs[st.session_state.iteration])
            
            # Wait briefly and then allow moving to the next stage
            time.sleep(0.1)  # Small delay to ensure the UI updates
            
            if st.button("읽기 완료", key="paragraph_read_button"):
                paragraph_viewed()
        
        elif st.session_state.stage == "novelty_survey":
            # Show the paragraph for reference
            st.subheader("텍스트:")
            st.write(st.session_state.paragraphs[st.session_state.iteration])
            
            # Novelty rating
            st.subheader("텍스트의 내용이 얼마나 생소한가요?")
            novelty_rating = st.radio(
                "텍스트의 생소함을 평가해주세요:",
                options=["1", "2", "3", "4", "5", "6", "7"],
                index=None,
                key=f"novelty_rating_{st.session_state.iteration}",
                help="1 = 전혀 생소하지 않음 (매우 익숙함), 7 = 매우 생소함 (완전히 새로움)",
                horizontal=True
            )
            
            # Store the rating immediately when selected
            if novelty_rating is not None:
                st.session_state.novelty_rating = novelty_rating
            
            # Difficulty rating
            st.subheader("텍스트의 내용이 얼마나 어려웠나요?")
            difficulty_rating = st.radio(
                "텍스트의 어려움을 평가해주세요:",
                options=["1", "2", "3", "4", "5", "6", "7"],
                index=None,
                key=f"difficulty_rating_{st.session_state.iteration}",
                help="1 = 전혀 어렵지 않음 (매우 쉬움), 7 = 매우 어려움 (이해하기 힘듦)",
                horizontal=True
            )
            
            # Store the rating immediately when selected
            if difficulty_rating is not None:
                st.session_state.difficulty_rating = difficulty_rating

            # Comments section
            paragraph_comments = st.text_area(
                "[파일럿용 피드백] 텍스트 생소함 및 난이도 관련 코멘트 (선택):",
                key=f"paragraph_comments_{st.session_state.iteration}",
                height=100
            )
            
            # Store comments immediately
            st.session_state.paragraph_comments = paragraph_comments
            
            if st.button("다음", key="novelty_submit_button"):
                submit_novelty_survey()
        
        elif st.session_state.stage == "ask_question":
            # Show paragraph again as reference
            st.subheader("텍스트:")
            st.write(st.session_state.paragraphs[st.session_state.iteration])
            
            # Show question input
            st.subheader("텍스트에 대해 떠오르는 질문을 적어주세요:")
            
            user_question = st.text_input(
                "질문 입력:", 
                key=f"user_question_{st.session_state.iteration}",
                on_change=lambda: log_textarea_focus("question_input") if not hasattr(st.session_state, 'question_input_focus_time') else None
            )
            
            # Store question immediately when typed
            st.session_state.user_question = user_question
            
            # Comments section
            question_comments = st.text_area(
                "[파일럿용 피드백] 질문 입력 관련 코멘트 (선택):",
                key=f"question_comments_{st.session_state.iteration}",
                height=100
            )
            
            # Store comments immediately
            st.session_state.question_comments = question_comments
            
            if st.button("질문 제출", key="question_submit_button"):
                submit_question()
        
        elif st.session_state.stage == "show_feedback":
            # Show paragraph again as reference
            st.subheader("텍스트:")
            st.write(st.session_state.paragraphs[st.session_state.iteration])
            
            # Show the question
            current_question = st.session_state.current_iteration_data.get('user_question', '')
            
            st.subheader("입력한 질문:")
            st.write(current_question if current_question else "Question not available")
            
            # Show AI feedback
            st.subheader("AI 피드백:")
            st.markdown("""아래는 AI가 연구 참여자의 질문에 대해 제시한 피드백입니다.""")
            current_feedback = st.session_state.current_iteration_data.get('feedback', '')
            st.markdown(f'**{current_feedback}**')
            
            # Comments section
            feedback_comments = st.text_area(
                "[파일럿용 피드백] AI 피드백 관련 코멘트 (선택):",
                key=f"feedback_comments_{st.session_state.iteration}",
                height=100
            )
            
            # Store comments immediately
            st.session_state.feedback_comments = feedback_comments

            # next step explanation
            st.markdown("""다음으로 넘어가면 AI 피드백에 대한 설문이 제시됩니다. AI 피드백을 완전히 숙지하고 넘어가주세요.""")
            
            if st.button("다음", key="feedback_next_button"):
                send_marker("survey_start")
                next_stage("survey")
        
        elif st.session_state.stage == "survey":
            # Get the current feedback condition
            current_paragraph = st.session_state.paragraphs[st.session_state.iteration]
            original_paragraph_index = st.session_state.paragraph_mapping.get(st.session_state.iteration, st.session_state.iteration)
            feedback_type = st.session_state.condition_mapping.get(original_paragraph_index, "no_feedback")
            
            # Show survey questions
            st.subheader("다음 설문 문항에 응답해주세요:")
            
            # Always show curiosity question
            curiosity_rating = st.radio(
                "AI의 피드백에 대해 얼마나 호기심을 느꼈나요?",
                options=["1", "2", "3", "4", "5", "6", "7"],
                index=None,
                key=f"curiosity_{st.session_state.iteration}",
                help="1 = 전혀 호기심을 느끼지 않음, 7 = 매우 호기심을 느낌",
                horizontal=True
            )
            
            # Store rating immediately
            if curiosity_rating is not None:
                st.session_state.curiosity = curiosity_rating
            
            # Only show relatedness questions if feedback_type is NOT "no_feedback"
            if feedback_type != "no_feedback":
                relatedness_rating = st.radio(
                    "AI의 피드백이 얼마나 자신의 질문과 관련되었나요?",
                    options=["1", "2", "3", "4", "5", "6", "7"],
                    index=None,
                    key=f"relatedness_{st.session_state.iteration}",
                    help="1 = 전혀 관련되지 않음, 7 = 매우 관련됨",
                    horizontal=True
                )
                
                # Store rating immediately
                if relatedness_rating is not None:
                    st.session_state.relatedness = relatedness_rating
                
            else:
                # For no_feedback condition, set these to None or default values
                st.session_state.relatedness = None

            accept_feedback_option = st.radio(
                "피드백을 수용할 의향이 있으신가요?",
                options=["예", "아니오"],
                index=None,
                key=f"accept_feedback_{st.session_state.iteration}",
            )
            
            # Store selection immediately
            if accept_feedback_option is not None:
                st.session_state.accept_feedback = accept_feedback_option
            
            # Comments section
            survey_comments = st.text_area(
                "[파일럿용 피드백] 설문이나 실험 관련 추가 코멘트 (선택):",
                key=f"survey_comments_{st.session_state.iteration}",
                height=100
            )
            
            # Store comments immediately
            st.session_state.survey_comments = survey_comments
            
            if st.button("설문 제출", key="survey_submit_button"):
                submit_survey()
        
        elif st.session_state.stage == "edit_question":
            # Show the original question and AI suggestion for reference
            current_question = st.session_state.current_iteration_data.get('user_question', '')
                
            st.subheader("텍스트:")
            st.write(st.session_state.paragraphs[st.session_state.iteration])
            
            st.subheader("입력한 질문:")
            st.write(current_question if current_question else "Question not available")
            
            st.subheader("AI 피드백:")
            current_feedback = st.session_state.current_iteration_data.get('feedback', '')
            st.markdown(current_feedback)
            
            # Allow editing the question
            st.subheader("질문 수정:")
            st.write("아래 영역에서 질문을 수정할 수 있습니다. 질문을 조금 더 창의적으로 바꾸어보세요.")
            
            # Use current question as initial value
            initial_value = current_question if current_question else ""
            
            edited_question = st.text_area(
                "수정된 질문:",
                value=initial_value,
                key=f"edited_question_{st.session_state.iteration}",
                height=100
            )
            
            # Store edited question immediately
            st.session_state.edited_question = edited_question
            
            # Comments section
            edit_comments = st.text_area(
                "[파일럿용 피드백] 질문 수정 관련 코멘트 (선택):",
                key=f"edit_comments_{st.session_state.iteration}",
                height=100
            )
            
            # Store comments immediately
            st.session_state.edit_comments = edit_comments

            st.write("최종 제출 버튼을 두 번 누르면 다음으로 넘어갑니다.")
            
            if st.button("최종 제출", key=f"final_submit_button_{st.session_state.iteration}"):
                submit_edited_question()

if __name__ == "__main__":
    main()
