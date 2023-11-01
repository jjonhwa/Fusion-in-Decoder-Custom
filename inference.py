# -*- coding: utf-8 -*-
import os
import argparse
from tqdm import tqdm

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch.nn.functional as F

from FiD.src import model
from FiD.src.modeling_t5 import T5ForConditionalGeneration
# from retrieval.model import BertEncoder_For_CrossEncoder

def get_top10(
        question,
        context,
        retrieval, 
        tokenizer, 
        n_context=10,
        token_length=200,
):

    tokenized_context = tokenizer(
        question,
        context,
        truncation='only_second',
        stride=int(token_length * 0.2),
        max_length=token_length,
        return_overflowing_tokens=True,
        padding='max_length',
        return_tensors='pt',
    )

    context_prob_pair = []
    count = len(tokenized_context['input_ids'])
    for i in range(count):
        input_ids = torch.tensor(tokenized_context['input_ids'][i].unsqueeze(dim=0))#.to(device)
        attention_mask = torch.tensor(tokenized_context['attention_mask'][i].unsqueeze(dim=0))#.to(device)
        token_type_ids = torch.tensor(tokenized_context['token_type_ids'][i].unsqueeze(dim=0))#.to(device)

        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")
            token_type_ids = token_type_ids.to("cuda")

        model_input = {"input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids}
        prob = retrieval(**model_input)
        prob = F.softmax(prob[0], dim=-1).detach().cpu().numpy()
        prob = prob[0][1]

        decoded_context = tokenizer.decode(input_ids[0])

        decoded_context = decoded_context.split("[SEP]")
        decoded_context = decoded_context[1].strip()

        context_prob_pair.append((decoded_context, prob))
            
    sorted_pair = sorted(context_prob_pair, key=lambda x: -x[1])
    
    sorted_context = [context for context, _ in sorted_pair]
    sorted_score = [score for _, score in sorted_pair]

    return sorted_context[:n_context], sorted_score[:n_context]

def get_ans(
    retrieved_passages,
    model,
    token_length=200,
    output_length=50,
):
    encoded_text = tokenizer.batch_encode_plus(
        retrieved_passages,
        max_length=token_length,
        pad_to_max_length=True,
        return_tensors='pt',
        truncation=True,
    )

    outputs = model.generate(
        input_ids = encoded_text['input_ids'].unsqueeze(0).to(device),
        attention_mask = encoded_text['attention_mask'].unsqueeze(0).to(device),
        max_length = output_length,
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def input_format(passages, question=None, type="QA"):
    if type == "QA":
        question = "question: " + question
        
        f = "context:" + " {}"
        passages = [f.format(c) for c in passages]
        passages = [question + " " + p for p in passages]
        return passages
    
    elif type == "SUMMARY":
        f = "서술하시오:" + " {}"
        passages = [f.format(c) for c in passages]
        return passages
    
    else:
        ValueError("Can just use 'QA' or 'SUMMARY'")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="KETI-AIR/ke-t5-large")
    parser.add_argument("--retrieval_path", type=str, default='jjonhwa/paragraph_s36000')
    parser.add_argument("--fid_path", type=str, default="/Users/choi/Desktop/글로랑/AGC_대회/AGC_second/FiD/save/SUMMARY_FiDT5_best_s27000_V2.pt")
    parser.add_argument("--question", action='store_true')
    parser.add_argument("--context", action='store_true')
    args = parser.parse_args()

    device = 'cpu'
    retrieval = AutoModelForSequenceClassification.from_pretrained(args.retrieval_path).to(device)
    retriever_tokenizer = AutoTokenizer.from_pretrained(args.retrieval_path)

    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_name)

    t5 = T5ForConditionalGeneration.from_pretrained(args.model_name)#, device_map="auto")
    FiD_model = model.FiDT5(t5.config, first_k=None)
    FiD_model = model.convert_LSA(FiD_model, n_cross_layer=6)
    FiD_model = nn.DataParallel(FiD_model)

    if torch.cuda.is_available():
        FiD_model.load_state_dict(torch.load(args.fid_path))
    else:
        FiD_model.load_state_dict(torch.load(args.fid_path, map_location=torch.device('cpu')))
    FiD_model = FiD_model.module if hasattr(FiD_model, 'module') else FiD_model
    FiD_model = FiD_model.to(device)

    if not args.question:
        args.question = '장기이식으로 인해 얻을 수 있는 효과는?'
    if not args.context:
        args.context = """
        장기 이식 - 위키백과, 우리 모두의 백과사전 장기 이식 위키백과, 우리 모두의 백과사전. 둘러보기로 가기 검색하러 가기 1967년 남아공에서 있었던 최초의 심장 이식 수술을 재연한 모습. 케이프타운 박물관. 장기 이식(臟器移植)이란 어떤 조직 또는 장기의 파손된 기능을 대체할 목적으로 원래 존재하는 장소에서 다른 장소로 조직 또는 장기를 옮기는 것이다. 장기 이식이 필요한 경우는 그 장기가 꼭 필요하고 대체할 수 없을 경우다. 위의 경우 위가 없어도 장으로 연결해서 사람이 살아갈 수 있다. 그러나 심장이나 간, 신장 등이 제 기능을 못하게 되었을 때는 이를 대체할 수 있는 것이 없기 때문에 장기 이식을 하지 않으면 죽게 된다. 심장과 신장 같은 경우는 대체 물품이 개발되어 있으나 한시적이고 실제적인 장기 역할이 힘들다. 신장이 안 좋은 사람은 매주 투석해야 하며, 투석을 하지 않을 경우 신부전증으로 진행되어 사망하게 된다. 간은 심장이나 신장과 달리 대체할 수 있는 것조차 없다. 현재의 기술 수준으로는 몸속에서 동일한 기능을 하는 심장, 간, 신장을 만들지 못하기 때문에 이식을 하게 된다. 참고로, 간이나 신장은 기증자가 살아있는 경우에도 기증이 가능하다. 간은 반으로 잘라내어도 생존할 수 있으며 시간이 지나면서 크기와 기능을 회복하게 된다. 그러나 수명이 수술로인해 어느정도까지 연장될수있는지는 사람마다 다르며 신장은 한 사람에게 두 개씩 있으며 한 개만으로도 살 수 있기는 하나 두개인 이유가 있기때문에 많이 불편할 수 있다. 판막같은 경우나 돼지로도 이식수술을 하는경우가 있다.그 이유는 사람과 가장 맞기 때문이라고 한다. 공여자가 없을 경우 꼭 생존자기증이 아니여도 대체할수 있는 정도로 할 수 있는 것들이 조금씩 늘어나고 있다. 목차 1 역사 2 종류 3 장기 이식의 성공 사례 4 윤리적 문제점 4.1 동종 장기이식의 문제점 4.2 이종 장기이식의 문제점 5 장기이식의 윤리적 고려사항 5.1 장기이식에 관한 기본 이념 5.2 장기적출금지대상 5.3 장기기증 요건 규정 6 장기이식 관리체계의 문제점 6.1 능동적 장기구득 체계 부재 6.2 장기구득 및 관리과정의 비효율성, 경직성 7 장기이식 관리체계의 개선해야할 점 7.1 장기이식 법률 개정 필요사항 7.2 향후 추진 과제 8 같이 보기 역사[편집] 상대적으로 긴 수술 역사를 가지는 성공적인 인간 동종 이식은 오랫동안 존재하였으며, 이후 수술 후의 생존에 대한 필요성이 발견되었다. 종류[편집] 자가이식(autograft)\xa0: 자신의 신체 일부를 특정한 곳에서 다른 곳으로 이식하는 장기 이식 동종이식(allograft)\xa0: 심장이나 신장 등과 같이 사람과 사람 간에 이루어지는 장기 이식 이종이식(xenograft)\xa0: 동물로부터 적출하여 인간에게 장기 이식 장기 이식의 성공 사례[편집] 장기 이식 (1968) 우리 나라 최초의 장기 이식은 1969년 가톨릭대 서울성모병원에서 이용각 교수가 집도한 신장 이식의 성공이 그 효시다. 그 후 1988년 뇌사자로부터 적출한 간 이식이 성공하면서 뇌사에 관한 사회적 관심을 불러일으켰고, 1992년 췌장 및 심장 이식이 성공한 이후 장기 이식이 본격화되었다. 한국 최초 뇌사자의 소장 이식 성공 (2009년 4월 15일) 가톨릭대 서울성모병원 장기이식센터 이명덕 교수(소아외과)팀은 2008년 12월 31일 위장관 손상으로 인해 단장증후군 상태에 있던 한송희(22, 여, 경기도 오산)에게 뇌사자의 소장을 이식했다. ‘물 풍선을 통한 복강 확장술’은 부작용이 없으며 가톨릭대 이명덕 교수팀에 의해 국내 최초로 성공하였고 한 차원 높은 소장 이식 수준을 국제적으로 인정받는 계기를 마련하였다. 에이즈 보균자간의 장기 이식 첫 성공 (2008년 10월 27일) 남아공 케이프타운의 한 병원에서 에이즈 바이러스(HIV) 보균자가 기부한 신장을 2명의 다른 에이즈 바이러스 보균자에 이식하는 수술이 성공했다. 이는 세계에서 처음으로 성공한 에이즈 바이러스 보균자 간의 장기 이식 수술로, 에이즈 바이러스 보균자의 생명 연장에 획기적인 성과로 평가받고 있다. 장기 이식용 미니돼지의 생산 성공 (2009년 4월 22일) 국내 연구진이 장기 이식이 필요한 환자에게 돼지의 장기를 면역거부가 거의 없이 이식할 수 있도록 해주는 복제돼지를 생산하는 데 성공했다. 미국에 이어 두 번째 성공으로, 이 연구 성과로 인해 세계 이종장기 시장 개척의 발판이 마련될 전망이다. 윤리적 문제점[편집] 동종 장기이식의 문제점[편집] 동종간 장기이식은 장기이식 중에서 가장 안정성이 높은 시술이지만 장기공여자(donor organ)의 부족이라는 문제점이 있다. 이는 세계 공통의 문제점으로, 그 예로 미국의 한해 장기이식 대기 환자 수는 6만 5천명으로 추정되는데, 그 수가 장기기증의 부족으로 매년 10-15%씩 증가하고 있다. 이는 우리나라의 상황도 크게 다르지 않은데, 연간 장기이식 대기자의 수는 2000년도에 3,730명에서 2005년 7,751명으로 2배 이상 증가했다. 증가하는 동종간 장기공여자의 부족현상으로 인해서 사체장기이식, 장기매매 등이 암암리에 행해지고 있다. 중국에서는 사형을 앞둔 사형수의 장기를 이식받는 방법으로 장기를 구하는 극단적 방법이 성행하고 있다. 이종 장기이식의 문제점[편집] 이종 장기이식의 가장 큰 위험성은 동물을 매개로 한 새로운 바이러스성 전염병의 발생이다. 동물 장기 이식은 1905년 이후 여러 차례 시도 되었다. 지금까지 전 세계적으로 82명이 침팬지, 원숭이, 돼지, 염소 등 동물의 장기와 조직을 이식받았으나 성공한 예는 극히 드물다. 동물의 장기를 사람에게 이식하면 인체의 면역체계가 항체를 만들어 이식 장기를 이물질로 인식하여 파괴하는 거부반응을 일으키기 때문에 이식한 장기나 부위가 괴사해 사망에 이른다. 따라서 이종 장기이식의 인체 적용은 충분한 연구가 필요하다. 전염병이 발생한 사례 돼지 독감\xa0: 바이러스 구조의 유사성으로 볼 때 돼지에게서 사람에게 전파된 것으로 추정. AIDS\xa0: 원숭이와 사람의 접촉에서 유래. 인간 광우병\xa0: 유럽에서 광우병에 걸린 쇠고기를 먹은 사람들이 병에 걸린다. 장기이식의 윤리적 고려사항[편집] 보건복지부는 1997년 8월 5일 ‘장기 등 이식에 관한 법률안’을 입법 예고하였다. 법률안은 뇌사도 사망의 한 형태로 인정, 현재 의료계가 자율적으로 실시하고 있는 뇌사자의 장기적출을 합법화하기로 하였다. 이로 인해 뇌사자의 장기공여에 의한 ‘장기이식수술’이라는 의료행위가 합법성을 획득하게 되었고, 장기이식의 수술건수가 급격히 증가하였다. 보건복지부가 제시하고 있는 법률안에는 장기공여자 의사 존중과 장기매매행위의 금지, 뇌사판정의 기준, 생명윤리위원회를 비롯한 관련 위원회의 지정 등에 관한 내용이 제시되어 있다. 장기기증자에 대한 금전적 보상 문제 장기이식 대기자와 수요에 비해 공급이 매우 적으므로 장기이식이 장기매매로 이어질 가능성이 높다. 미국에서는 1984 년 국립 장기 이식법 (National Organ Transplant Act)에 의해 장기 판매가 불법이 되었다. 영국에서는 1989 년 인간 장기 이식법 (Human Organ Transplans Act 1989)이 처음으로 기관 판매를 불법으로 만들었다. 그러나 이제 장기 기증자에 대한 금전적 보상이 호주에서 합법화되고 있으며 싱가포르의 경우 신장 이식의 경우에만 엄격히 적용된다. 기부자는 보상 기부금으로 장기 대신에 돈이나 기타 보상을받습니다. 이러한 관행은 합법적이든 아니든 간에 세계의 일부 지역에서 흔히 볼 수 있으며 의료 관광을 주도하는 여러 요소 중 하나가 되고 있다. 일부 사람들은 장기 거래의 활성화가 장기 수요 부족 문제점을 해결할 수 있다고 생각한다. 게리 베커 (Gary Becker)와 훌리오 엘리아 (Julio Elias)의 "생체 및 사체 장기 기증을위한 시장에서 인센티브 도입"에 대한 기사는 자유 시장이 장기 이식에서의 희소성 문제를 해결할 수 있다고 말했다. Mark Cherry (조지 타운 대학 출판사, 2005)의 두 권의 책, 스테이크와 신장\xa0: 제임스 스테이시 테일러 (James Stacey Taylor)는 왜 인체 부분의 시장이 도덕적으로 필수적인가\xa0: (Ashgate Press, 2005); 시장을 이용하여 장기 이식을 위한 장기 공급을 늘려라. 등을 펴내며 장기판매를 지지한다. 이란은 1988 년 이래로 신장을위한 법적 시장을 가지고 있다. 기부자는 정부에 의해 약 미화 1200 달러가 지급되며 수혜자 또는 지역 자선 단체에서 추가로 기금을 수령한다. Economist 와 Ayn Rand Institute는 합법적인 시장을 승인하고 옹호한다. 그들은 19 세에서 65 세 사이의 미국인 중 0.06 %가 신장 하나를 판매한다면 국가 대기자 명단이 사라질 것이라고 주장했다. 장기 이식이 장기 공급 부족의 문제를 해결할 수 있다. 하지만 장기를 돈으로 물건처럼 쉽게 사고 팔수 있도록 허용하는 것이 과연 옳은 것인지, 이후의 윤리적 문제점이 있지는 않은지에 대한 윤리적 쟁점이 논의되고 있다. 장기이식에 관한 기본 이념[편집] 인도적 정신에 따라 [기증자]의 자발적인 의사를 존중하고, 이식을 필요로 하는 자에게 [공평]한 기회제공. 적출 또는 이식대상 장기 등의 범위 규정. 장기의 [매매] 또는 매매 교사, [알선] 금지. 서류 위조 금지 장기적출금지대상[편집] 살아있는자 중 16세 미만인자. [정신질환자]. [정신지체인]. [마약], [대마] 및 향정신성약품에 중독된 자. 중추신경계 질환 기타 감염병 또는 약물투여자 장기기증 요건 규정[편집] 생전에 장기기증에 동의한 경우로서 가족 또는 유족이 명시적으로 거부하지 아니하는 경우가 아닌 장기기증을 희망하는 자 그렇다 하더라도 강요되어서도 안되며 희망해서 신청을 했다가도 마음을 돌릴수도 있음. 의사 존중 본인의 기증의사가 확인되지 아니한 경우로써 그 가족 또는 유족이 동의하는 경우. 장기이식 관리체계의 문제점[편집] 능동적 장기구득 체계 부재[편집] 잠재뇌사자 기증의사확인 및 설득체계가 확립되어있지 못하다. 장기구득 및 관리과정의 비효율성, 경직성[편집] 장기이식관리기관의 역할이 아직 미흡하다. 장기이식 관리체계의 개선해야할 점[편집] 장기이식 법률 개정 필요사항[편집] 법에 규정된 장기 이외 이식을 할 경우 사전 승인. 장기기증의사 확인 의무화. 장기이식후 이식결과 보고. 장기이식후 부작용 신고 및 조사체계 도입. 향후 추진 과제[편집] 생전 장기기증 의사결정 및 표시 기회 확산. -[운전면허증], [주민등록법|주민등록증], [건강보험증]. Donor Hospital 활성화. -뇌사추정자 신고 및 기증의사확인 제도 정착. -뇌사자 관리 강화. 장기구득기관 도입 -> 능동적 장기구득체계. [KONOS]기능 및 지원체계 강화. 장기이식 안전성 강화. -사전검사강화, 부작용 보고 및 조사체계 등. 장기이식 관련 정보시스템 구축. 같이 보기[편집] 위키미디어 공용에 관련된미디어 분류가 있습니다.장기 이식 인공 장기 신장 이식(en) 전거 통제 LCCN: sh85137008 GND: 4060675-2 원본 주소 "https://ko.wikipedia.org/w/index.php?title=장기_이식&oldid=24262708" 분류: 장기 이식숨은 분류: LCCN 식별자를 포함한 위키백과 문서GND 식별자를 포함한 위키백과 문서 둘러보기 메뉴 개인 도구 로그인하지 않음토론기여계정 만들기로그인 이름공간 문서토론 변수 보기 읽기편집역사 보기 더 보기 검색 둘러보기 대문사용자 모임요즘 화제최근 바뀜모든 문서 보기임의 문서로도움말기부 도구 여기를 가리키는 문서가리키는 글의 최근 바뀜파일 올리기특수 문서 목록고유 링크문서 정보위키데이터 항목이 문서 인용하기 다른 프로젝트 위키미디어 공용 인쇄/내보내기 책 만들기PDF로 다운로드인쇄용 판 다른 언어 AfrikaansالعربيةAzərbaycancaБеларускаяБългарскиBosanskiCatalàČeštinaCymraegDanskDeutschΕλληνικάEnglishEsperantoEspañolEestiEuskaraفارسیSuomiFrançaisGaeilgeGalego客家語/Hak-kâ-ngîעבריתहिन्दीHrvatskiMagyarՀայերենInterlinguaBahasa IndonesiaÍslenskaItaliano日本語ქართულიҚазақшаКыргызчаLatinaLatviešuМакедонскиമലയാളംBahasa MelayuPlattdüütschनेपालीनेपाल भाषाNederlandsNorsk nynorskNorskNouormandपालिPolskiPortuguêsRomânăРусскийसंस्कृतम्Srpskohrvatski / српскохрватскиSimple EnglishSlovenčinaСрпски / srpskiSvenskaதமிழ்ไทยTagalogTürkçeУкраїнськаOʻzbekcha/ўзбекчаTiếng Việtייִדיש中文粵語 링크 편집 이 문서는 2019년 5월 23일 (목) 09:22에 마지막으로 편집되었습니다. 모든 문서는 크리에이티브 커먼즈 저작자표시-동일조건변경허락 3.0에 따라 사용할 수 있으며, 추가적인 조건이 적용될 수 있습니다. 자세한 내용은 이용 약관을 참고하십시오.Wikipedia®는 미국 및 다른 국가에 등록되어 있는 Wikimedia Foundation, Inc. 소유의 등록 상표입니다. 개인정보 정책 위키백과 소개 면책 조항 개발자 쿠키 정책 모바일 보기
        """
    
    retrieved_context, _ = get_top10(
        args.question,
        args.context,
        retrieval,
        retriever_tokenizer,
    )

    passages = input_format(
        passages=retrieved_context,
        question=args.question,
        type='QA'
    )
    
    answer = get_ans(
        retrieved_context,
        FiD_model,
        output_length=100
    )

    print(f"정답: {answer}")