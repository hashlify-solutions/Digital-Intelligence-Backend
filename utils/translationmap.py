topic_translation_map = {
    "النقاش السياسي": "political_discussion",
    "التعليق الاجتماعي": "social_commentary",
    "ردود الفعل على الأخبار": "news_reaction",
    "المحتوى الديني": "religious_content",
    "المناقشة الثقافية": "cultural_discussion",
    "السياسة الإقليمية": "regional_politics",
    "العلاقات الدولية": "international_relations",
    "دعم المجتمع": "community_support",
    "التظلم": "grievance",
    "الاحتفال": "celebration",
    "التضامن": "solidarity",
    "الانتقاد": "criticism",
    "الوطنية": "patriotism"
}

interaction_translation_map = {
    "التعبير عن الدعم": "support_expression",
    "التعبير عن المعارضة": "opposition_expression",
    "ملاحظة محايدة": "neutral_observation",
    "استجابة عاطفية": "emotional_response",
    "دعوة للعمل": "call_to_action",
    "مشاركة المعلومات": "information_sharing",
    "مشاركة الرأي": "opinion_sharing",
    "اتفاق": "agreement",
    "اختلاف": "disagreement",
    "التساؤل": "questioning"
}

sentiment_translation_map = {
    "وطني": "patriotic",
    "قومي": "nationalistic",
    "داعم": "supportive",
    "انتقادي": "critical",
    "دفاعي": "defensive",
    "عدواني": "aggressive",
    "سلمي": "peaceful",
    "تصالحي": "reconciliatory",
    "انقسامي": "divisive",
    "موحد": "unifying",
    "ديني": "religious",
    "علماني": "secular"
}

sentiments_default = [
                "patriotic", "nationalistic", "supportive", "critical",
                "defensive", "aggressive", "peaceful", "reconciliatory",
                "divisive", "unifying", "religious", "secular"
            ]

interactions_default = [
                "support_expression", "opposition_expression", "neutral_observation",
                "emotional_response", "call_to_action", "information_sharing",
                "opinion_sharing", "agreement", "disagreement", "questioning"
]

topics_default = ["political_discussion", "social_commentary", "news_reaction",
                "religious_content", "cultural_discussion", "regional_politics",
                "international_relations", "community_support", "grievance",
                "celebration", "solidarity", "criticism", "patriotism"]

note_classification_default = ["Social Media", "Banking", "Shopping", "News Sites", "Entertainment", "Educational", "Government Sites", "Healthcare Sites", "Travel Sites", "Gaming Sites", "Adult Content", "Political Sites"]

browsing_history_classification_default = ["Password", "Username", "Personal Information", "Financial Data", "Contact Details", "Medical Information", "Legal Documents", "Credentials", "API Keys", "Security Notes", "Private Thoughts", "Work Notes", "Personal Notes", "Confidential Information", "Sensitive Data"]

sentiments_arabic = [
    "وطني",
    "قومي",
    "داعم",
    "انتقادي",
    "دفاعي",
    "عدواني",
    "سلمي",
    "تصالحي",
    "انقسامي",
    "موحد",
    "ديني",
    "علماني"
]

interactions_arabic = [
    "التعبير عن الدعم",
    "التعبير عن المعارضة",
    "ملاحظة محايدة",
    "استجابة عاطفية",
    "دعوة للعمل",
    "مشاركة المعلومات",
    "مشاركة الرأي",
    "اتفاق",
    "اختلاف",
    "التساؤل"
]

topics_arabic = [
    "النقاش السياسي",
    "التعليق الاجتماعي",
    "ردود الفعل على الأخبار",
    "المحتوى الديني",
    "المناقشة الثقافية",
    "السياسة الإقليمية",
    "العلاقات الدولية",
    "دعم المجتمع",
    "التظلم",
    "الاحتفال",
    "التضامن",
    "الانتقاد",
    "الوطنية"
]

entitiesClasses_arabic = [
    "شخص",           # person
    "منظمة",         # organization
    "موقع",          # location
    "مجموعة",        # group
    "حدث",           # event
    "تاريخ",         # date
    "وقت",           # time
    "رقم",           # number
    "عملة",          # currency
    "منتج",          # product
    "هاشتاج",        # hashtag
    "وسائل_إعلام",   # media
    "مؤسسة_حكومية",  # government entity
    "مدينة",         # city
    "دولة",          # country
    "علامة_تجارية",   # brand
    "قضية",          # issue/case
    "مناسبة",        # occasion
]

entitiesClasses_default = [
    "person",
    "organization",
    "location",
    "group",
    "event",
    "date",
    "time",
    "number",
    "currency",
    "product",
    "hashtag",
    "media",
    "government_entity",
    "city",
    "country",
    "brand",
    "case",
    "occasion"
]

# Minimal default entity classes to align with UI top entities
entitiesClasses_ui_en = [
    "Person",
    "Organization",
    "Brand",
    "Location",
    "Number"
]

entitiesClasses_ui_ar = [
    "شخص",
    "منظمة",
    "علامة_تجارية",
    "موقع",
    "رقم"
]