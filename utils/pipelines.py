import copy

sankey_flow_topic_to_risk_level_pipeline=[
    {
        '$group': {
            '_id': {
                'topic': {
                    '$ifNull': [
                        '$analysis_summary.top_topic', 'Unknown'
                    ]
                }, 
                'risk_level': {
                    '$ifNull': [
                        '$analysis_summary.risk_level', 'None'
                    ]
                }
            }, 
            'count': {
                '$sum': 1
            }
        }
    }, {
        '$group': {
            '_id': None, 
            'nodes': {
                '$addToSet': '$_id.topic'
            }, 
            'targets': {
                '$addToSet': '$_id.risk_level'
            }, 
            'links': {
                '$push': {
                    'source': '$_id.topic', 
                    'target': '$_id.risk_level', 
                    'value': '$count'
                }
            }
        }
    }, {
        '$project': {
            '_id': 0, 
            'nodes': {
                '$concatArrays': [
                    '$nodes', '$targets'
                ]
            }, 
            'links': 1
        }
    }
]

heatmap_of_appilication_against_emotion_piepline=[
    {
        '$match': {
            'Application': {
                '$exists': True
            }
        }
    }, {
        '$addFields': {
            'Application': {
                '$cond': {
                    'if': {
                        '$or': [
                            {
                                '$eq': [
                                    '$Application', None
                                ]
                            }, {
                                '$not': [
                                    '$Application'
                                ]
                            }, {
                                '$eq': [
                                    {
                                        '$type': '$Application'
                                    }, 'double'
                                ]
                            }
                        ]
                    }, 
                    'then': 'Unknown', 
                    'else': '$Application'
                }
            }
        }
    }, {
        '$group': {
            '_id': {
                'application': '$Application', 
                'emotion': {
                    '$ifNull': [
                        '$analysis_summary.emotion', 'Unknown'
                    ]
                }
            }, 
            'count': {
                '$sum': 1
            }
        }
    }, {
        '$group': {
            '_id': '$_id.application', 
            'emotions': {
                '$push': {
                    'emotion': '$_id.emotion', 
                    'count': '$count'
                }
            }
        }
    }, {
        '$project': {
            '_id': 0, 
            'application': '$_id', 
            'emotions': 1
        }
    }
]

bar_and_pipe_chart_pipeline= [
    {
        '$facet': {
            'topics': [
                {
                    '$group': {
                        '_id': {
                            '$ifNull': [
                                '$analysis_summary.top_topic', 'others'
                            ]
                        }, 
                        'count': {
                            '$sum': 1
                        }
                    }
                }, {
                    '$sort': {
                        'count': -1
                    }
                }
            ], 
            'sentiment_aspects': [
                {
                    '$group': {
                        '_id': {
                            '$ifNull': [
                                '$analysis_summary.sentiment_aspect', 'others'
                            ]
                        }, 
                        'count': {
                            '$sum': 1
                        }
                    }
                }, {
                    '$sort': {
                        'count': -1
                    }
                }
            ], 
            'interaction_types': [
                {
                    '$group': {
                        '_id': {
                            '$ifNull': [
                                '$analysis_summary.interaction_type', 'others'
                            ]
                        }, 
                        'count': {
                            '$sum': 1
                        }
                    }
                }, {
                    '$sort': {
                        'count': -1
                    }
                }
            ], 
            'emotions': [
                {
                    '$group': {
                        '_id': {
                            '$ifNull': [
                                '$analysis_summary.emotion', 'others'
                            ]
                        }, 
                        'count': {
                            '$sum': 1
                        }
                    }
                }, {
                    '$sort': {
                        'count': -1
                    }
                }
            ], 
            'languages': [
                {
                    '$group': {
                        '_id': {
                            '$ifNull': [
                                '$analysis_summary.language', 'others'
                            ]
                        }, 
                        'count': {
                            '$sum': 1
                        }
                    }
                }, {
                    '$sort': {
                        'count': -1
                    }
                }
            ], 
            'risk_levels': [
                {
                    '$group': {
                        '_id': {
                            '$ifNull': [
                                '$analysis_summary.risk_level', 'others'
                            ]
                        }, 
                        'count': {
                            '$sum': 1
                        }
                    }
                }, {
                    '$sort': {
                        'count': -1
                    }
                }
            ], 
            'applications': [
                {
                    '$group': {
                        '_id': {
                            '$ifNull': [
                                '$Application', 'others'
                            ]
                        }, 
                        'count': {
                            '$sum': 1
                        }
                    }
                }, {
                    '$sort': {
                        'count': -1
                    }
                }
            ]
        }
    }
]

stacked_bar_risk_per_lanaguage=[
    {
        '$group': {
            '_id': {
                'language': {
                    '$ifNull': [
                        '$analysis_summary.language', 'Unknown'
                    ]
                }, 
                'risk_level': {
                    '$ifNull': [
                        '$analysis_summary.risk_level', 'None'
                    ]
                }
            }, 
            'count': {
                '$sum': 1
            }
        }
    }, {
        '$group': {
            '_id': '$_id.language', 
            'risk_level_per_language': {
                '$push': {
                    'risk_level': '$_id.risk_level', 
                    'count': '$count'
                }
            }
        }
    }, {
        '$sort': {
            '_id': 1
        }
    }
]

top_card_data_pipeline=[
    {
        '$facet': {
            'main_metrics': [
                {
                    '$group': {
                        '_id': None, 
                        'totalMessages': {
                            '$sum': 1
                        }, 
                        'highRiskMessages': {
                            '$sum': {
                                '$cond': [
                                    {
                                        '$eq': [
                                            '$analysis_summary.risk_level', 'high'
                                        ]
                                    }, 1, 0
                                ]
                            }
                        }, 
                        'uniqueUsers': {
                            '$addToSet': '$From'
                        }, 
                        'alertMessages': {
                            '$sum': {
                                '$cond': [
                                    {
                                        '$eq': [
                                            '$alert', True
                                        ]
                                    }, 1, 0
                                ]
                            }
                        }
                    }
                }, {
                    '$project': {
                        '_id': 0, 
                        'totalMessages': 1,
                        'highRiskMessages': 1, 
                        'uniqueUsers': {
                            '$size': '$uniqueUsers'
                        }, 
                        'alertMessages': 1
                    }
                }
            ],
            'applications_count': [
                {
                    '$group': {
                        '_id': {
                            '$ifNull': ['$Application', 'Unknown']
                        },
                        'count': { '$sum': 1 }
                    }
                },
                {
                    '$sort': { 'count': -1 }
                },
                {
                    '$limit': 3
                },
                {
                    '$group': {
                        '_id': None,
                        'applications': {
                            '$push': {
                                'k': '$_id',
                                'v': '$count'
                            }
                        }
                    }
                },
                {
                    '$project': {
                        '_id': 0,
                        'applications': { '$arrayToObject': '$applications' }
                    }
                }
            ],
            'entities_classes_count': [
                {
                    '$unwind': {
                        'path': '$analysis_summary.entities_classification',
                        'preserveNullAndEmptyArrays': False
                    }
                },
                {
                    '$project': {
                        'entity_class': { '$objectToArray': '$analysis_summary.entities_classification' }
                    }
                },
                {
                    '$unwind': '$entity_class'
                },
                {
                    '$group': {
                        '_id': '$entity_class.k',
                        'count': { '$sum': 1 }
                    }
                },
                {
                    '$sort': { 'count': -1 }
                },
                {
                    '$limit': 3
                },
                {
                    '$group': {
                        '_id': None,
                        'entities_classes': {
                            '$push': {
                                'k': '$_id',
                                'v': '$count'
                            }
                        }
                    }
                },
                {
                    '$project': {
                        '_id': 0,
                        'entities_classes': { '$arrayToObject': '$entities_classes' }
                    }
                }
            ]
        }
    },
    {
        '$project': {
            'totalMessages': { '$arrayElemAt': ['$main_metrics.totalMessages', 0] },
            'highRiskMessages': { '$arrayElemAt': ['$main_metrics.highRiskMessages', 0] },
            'uniqueUsers': { '$arrayElemAt': ['$main_metrics.uniqueUsers', 0] },
            'alertMessages': { '$arrayElemAt': ['$main_metrics.alertMessages', 0] },
            'top_3_applications_message_count': { '$arrayElemAt': ['$applications_count.applications', 0] },
            'top_3_entities_classes': { '$arrayElemAt': ['$entities_classes_count.entities_classes', 0] }
        }
    }
]

side_card_data_pipeline=[
    {
        '$facet': {
            'top_users': [
                {
                    '$group': {
                        '_id': '$From', 
                        'messageCount': {
                            '$sum': 1
                        }
                    }
                }, {
                    '$sort': {
                        'messageCount': -1
                    }
                }, {
                    '$limit': 5
                }
            ], 
            
            # 'last_5_messages': [
            #     {
            #         '$sort': {
            #             'ingestion_timestamp': -1
            #         }
            #     }, {
            #         '$limit': 5
            #     }, {
            #         '$addFields': {
            #             '_id': {
            #                 '$toString': '$_id'
            #             }, 
            #             'case_id': {
            #                 '$toString': '$case_id'
            #             }
            #         }
            #     }
            # ], 
            
            # 'messages_by_language': [
            #     {
            #         '$group': {
            #             '_id': '$analysis_summary.language', 
            #             'count': {
            #                 '$sum': 1
            #             }
            #         }
            #     }
            # ], 
            
            # 'messages_by_risk_evel': [
            #     {
            #         '$group': {
            #             '_id': '$analysis_summary.risk_level', 
            #             'count': {
            #                 '$sum': 1
            #             }
            #         }
            #     }
            # ], 
            
            # 'most_occurring_entities': [
            #     {
            #         '$unwind': '$analysis_summary.entities'
            #     }, {
            #         '$group': {
            #             '_id': '$analysis_summary.entities', 
            #             'count': {
            #                 '$sum': 1
            #             }
            #         }
            #     }, {
            #         '$sort': {
            #             'count': -1
            #         }
            #     }, {
            #         '$limit': 10
            #     }
            # ]
            
            'top_entities_by_category': [
                {
                    '$match': {
                        'analysis_summary.entities_classification': { '$exists': True }
                    }
                },
                {
                    '$project': {
                        'entity_categories': { '$objectToArray': '$analysis_summary.entities_classification' }
                    }
                },
                {
                    '$unwind': '$entity_categories'
                },
                {
                    '$project': {
                        'category': '$entity_categories.k',
                        'entities': '$entity_categories.v'
                    }
                },
                {
                    '$unwind': '$entities'
                },
                {
                    '$group': {
                        '_id': {
                            'category': '$category',
                            'entity': '$entities'
                        },
                        'count': { '$sum': 1 }
                    }
                },
                {
                    '$sort': {
                        'count': -1
                    }
                },
                {
                    '$group': {
                        '_id': '$_id.category',
                        'entities': {
                            '$push': {
                                'entity': '$_id.entity',
                                'count': '$count'
                            }
                        }
                    }
                },
                {
                    '$project': {
                        '_id': 0,
                        'category': '$_id',
                        'entities': 1
                    }
                }
            ]
        }
    }
]

area_chart_entities_pipeline=[
    {
        '$unwind': '$analysis_summary.entities'
    }, {
        '$group': {
            '_id': '$analysis_summary.entities', 
            'count': {
                '$sum': 1
            }
        }
    }, {
        '$sort': {
            'count': -1
        }
    }, {
        '$limit': 10
    }
]

entities_classes_vs_emotion_pipeline = [
    {
        '$unwind': {
            'path': '$analysis_summary.entities_classification',
            'preserveNullAndEmptyArrays': False
        }
    }, {
        '$project': {
            'entity_classes': { '$objectToArray': '$analysis_summary.entities_classification' },
            'emotion': { '$ifNull': ['$analysis_summary.emotion', 'Unknown'] }
        }
    }, {
        '$unwind': '$entity_classes'
    }, {
        '$group': {
            '_id': {
                'entity_class': '$entity_classes.k',
                'emotion': '$emotion'
            },
            'count': { '$sum': 1 }
        }
    }, {
        '$group': {
            '_id': '$_id.entity_class',
            'emotions': {
                '$push': {
                    'emotion': '$_id.emotion',
                    'count': '$count'
                }
            }
        }
    }, {
        '$project': {
            '_id': 0,
            'entity_class': '$_id',
            'emotions': 1
        }
    }
]

applications_vs_topics_pipeline = [
    {
        '$match': {
            'Application': { '$exists': True }
        }
    }, {
        '$addFields': {
            'Application': {
                '$cond': {
                    'if': {
                        '$or': [
                            { '$eq': ['$Application', None] },
                            { '$not': ['$Application'] },
                            { '$eq': [{ '$type': '$Application' }, 'double'] }
                        ]
                    },
                    'then': 'Unknown',
                    'else': '$Application'
                }
            }
        }
    }, {
        '$group': {
            '_id': {
                'application': '$Application',
                'topic': { '$ifNull': ['$analysis_summary.top_topic', 'Unknown'] }
            },
            'count': { '$sum': 1 }
        }
    }, {
        '$group': {
            '_id': '$_id.application',
            'topics': {
                '$push': {
                    'topic': '$_id.topic',
                    'count': '$count'
                }
            }
        }
    }, {
        '$project': {
            '_id': 0,
            'application': '$_id',
            'topics': 1
        }
    }
]

topics_vs_emotions_pipeline = [
    {
        '$group': {
            '_id': {
                'topic': { '$ifNull': ['$analysis_summary.top_topic', 'Unknown'] },
                'emotion': { '$ifNull': ['$analysis_summary.emotion', 'Unknown'] }
            },
            'count': { '$sum': 1 }
        }
    }, {
        '$group': {
            '_id': '$_id.topic',
            'emotions': {
                '$push': {
                    'emotion': '$_id.emotion',
                    'count': '$count'
                }
            }
        }
    }, {
        '$project': {
            '_id': 0,
            'topic': '$_id',
            'emotions': 1
        }
    }
]

entities_vs_emotions_pipeline = [
    {
        '$unwind': {
            'path': '$analysis_summary.entities',
            'preserveNullAndEmptyArrays': False
        }
    }, {
        '$group': {
            '_id': {
                'entity': '$analysis_summary.entities',
                'emotion': { '$ifNull': ['$analysis_summary.emotion', 'Unknown'] }
            },
            'count': { '$sum': 1 }
        }
    }, {
        '$sort': {
            'count': -1
        }
    }, {
        '$group': {
            '_id': '$_id.entity',
            'emotions': {
                '$push': {
                    'emotion': '$_id.emotion',
                    'count': '$count'
                }
            },
            'totalCount': { '$sum': '$count' }
        }
    }, {
        '$sort': {
            'totalCount': -1
        }
    }, {
        '$limit': 10
    }, {
        '$project': {
            '_id': 0,
            'entity': '$_id',
            'emotions': 1
        }
    }
]

entities_vs_applications_pipeline = [
    {
        '$match': {
            'Application': { '$exists': True },
            'analysis_summary.entities': { '$exists': True }
        }
    }, {
        '$addFields': {
            'Application': {
                '$cond': {
                    'if': {
                        '$or': [
                            { '$eq': ['$Application', None] },
                            { '$not': ['$Application'] },
                            { '$eq': [{ '$type': '$Application' }, 'double'] }
                        ]
                    },
                    'then': 'Unknown',
                    'else': '$Application'
                }
            }
        }
    }, {
        '$unwind': {
            'path': '$analysis_summary.entities',
            'preserveNullAndEmptyArrays': False
        }
    }, {
        '$group': {
            '_id': {
                'entity': '$analysis_summary.entities',
                'application': '$Application'
            },
            'count': { '$sum': 1 }
        }
    }, {
        '$sort': {
            'count': -1
        }
    }, {
        '$group': {
            '_id': '$_id.entity',
            'applications': {
                '$push': {
                    'application': '$_id.application',
                    'count': '$count'
                }
            },
            'totalCount': { '$sum': '$count' }
        }
    }, {
        '$sort': {
            'totalCount': -1
        }
    }, {
        '$limit': 10
    }, {
        '$project': {
            '_id': 0,
            'entity': '$_id',
            'applications': 1
        }
    }
]

entities_vs_topics_pipeline = [
    {
        '$unwind': {
            'path': '$analysis_summary.entities',
            'preserveNullAndEmptyArrays': False
        }
    }, {
        '$group': {
            '_id': {
                'entity': '$analysis_summary.entities',
                'topic': { '$ifNull': ['$analysis_summary.top_topic', 'Unknown'] }
            },
            'count': { '$sum': 1 }
        }
    }, {
        '$sort': {
            'count': -1
        }
    }, {
        '$group': {
            '_id': '$_id.entity',
            'topics': {
                '$push': {
                    'topic': '$_id.topic',
                    'count': '$count'
                }
            },
            'totalCount': { '$sum': '$count' }
        }
    }, {
        '$sort': {
            'totalCount': -1
        }
    }, {
        '$limit': 10
    }, {
        '$project': {
            '_id': 0,
            'entity': '$_id',
            'topics': 1
        }
    }
]

def entity_class_vs_emotions_pipeline(entity_class):
    """Pipeline to get top 10 entities of a specific class vs emotions"""
    return [
        {
            '$match': {
                f'analysis_summary.entities_classification.{entity_class}': {
                    '$exists': True,
                    '$ne': []
                }
            }
        }, {
            '$unwind': {
                'path': f'$analysis_summary.entities_classification.{entity_class}',
                'preserveNullAndEmptyArrays': False
            }
        }, {
            '$group': {
                '_id': {
                    'entity': f'$analysis_summary.entities_classification.{entity_class}',
                    'emotion': { '$ifNull': ['$analysis_summary.emotion', 'Unknown'] }
                },
                'count': { '$sum': 1 }
            }
        }, {
            '$sort': {
                'count': -1
            }
        }, {
            '$group': {
                '_id': '$_id.entity',
                'emotions': {
                    '$push': {
                        'emotion': '$_id.emotion',
                        'count': '$count'
                    }
                },
                'totalCount': { '$sum': '$count' }
            }
        }, {
            '$sort': {
                'totalCount': -1
            }
        }, {
            '$limit': 10
        }, {
            '$project': {
                '_id': 0,
                'entity': '$_id',
                'emotions': 1,
                'entity_class': { '$literal': entity_class }
            }
        }
    ]

def entity_class_vs_applications_pipeline(entity_class):
    """Pipeline to get top 10 entities of a specific class vs applications"""
    return [
        {
            '$match': {
                'Application': { '$exists': True },
                f'analysis_summary.entities_classification.{entity_class}': {
                    '$exists': True,
                    '$ne': []
                }
            }
        }, {
            '$addFields': {
                'Application': {
                    '$cond': {
                        'if': {
                            '$or': [
                                { '$eq': ['$Application', None] },
                                { '$not': ['$Application'] },
                                { '$eq': [{ '$type': '$Application' }, 'double'] }
                            ]
                        },
                        'then': 'Unknown',
                        'else': '$Application'
                    }
                }
            }
        }, {
            '$unwind': {
                'path': f'$analysis_summary.entities_classification.{entity_class}',
                'preserveNullAndEmptyArrays': False
            }
        }, {
            '$group': {
                '_id': {
                    'entity': f'$analysis_summary.entities_classification.{entity_class}',
                    'application': '$Application'
                },
                'count': { '$sum': 1 }
            }
        }, {
            '$sort': {
                'count': -1
            }
        }, {
            '$group': {
                '_id': '$_id.entity',
                'applications': {
                    '$push': {
                        'application': '$_id.application',
                        'count': '$count'
                    }
                },
                'totalCount': { '$sum': '$count' }
            }
        }, {
            '$sort': {
                'totalCount': -1
            }
        }, {
            '$limit': 10
        }, {
            '$project': {
                '_id': 0,
                'entity': '$_id',
                'applications': 1,
                'entity_class': { '$literal': entity_class }
            }
        }
    ]

def entity_class_vs_topics_pipeline(entity_class):
    """Pipeline to get top 10 entities of a specific class vs topics"""
    return [
        {
            '$match': {
                f'analysis_summary.entities_classification.{entity_class}': {
                    '$exists': True,
                    '$ne': []
                }
            }
        }, {
            '$unwind': {
                'path': f'$analysis_summary.entities_classification.{entity_class}',
                'preserveNullAndEmptyArrays': False
            }
        }, {
            '$group': {
                '_id': {
                    'entity': f'$analysis_summary.entities_classification.{entity_class}',
                    'topic': { '$ifNull': ['$analysis_summary.top_topic', 'Unknown'] }
                },
                'count': { '$sum': 1 }
            }
        }, {
            '$sort': {
                'count': -1
            }
        }, {
            '$group': {
                '_id': '$_id.entity',
                'topics': {
                    '$push': {
                        'topic': '$_id.topic',
                        'count': '$count'
                    }
                },
                'totalCount': { '$sum': '$count' }
            }
        }, {
            '$sort': {
                'totalCount': -1
            }
        }, {
            '$limit': 10
        }, {
            '$project': {
                '_id': 0,
                'entity': '$_id',
                'topics': 1,
                'entity_class': { '$literal': entity_class }
            }
        }
    ]

#filtered
def get_filtered_pipelines(ids:list):
    match_obj = {
        '$match': {
            '_id': {
                '$in': ids
            }
        }
    }
    sankey_flow_topic_to_risk_level_pipeline.insert(0,match_obj)
    heatmap_of_appilication_against_emotion_piepline.insert(0,match_obj)
    bar_and_pipe_chart_pipeline.insert(0,match_obj)
    stacked_bar_risk_per_lanaguage.insert(0,match_obj)
    top_card_data_pipeline.insert(0,match_obj)
    side_card_data_pipeline.insert(0,match_obj)
    area_chart_entities_pipeline.insert(0,match_obj)
    
    return {
        "sankey_flow_topic_to_risk_level_pipeline": [match_obj] + copy.deepcopy(sankey_flow_topic_to_risk_level_pipeline),
        "heatmap_of_appilication_against_emotion_piepline": [match_obj] + copy.deepcopy(heatmap_of_appilication_against_emotion_piepline),
        "bar_and_pipe_chart_pipeline": [match_obj] + copy.deepcopy(bar_and_pipe_chart_pipeline),
        "stacked_bar_risk_per_lanaguage": [match_obj] + copy.deepcopy(stacked_bar_risk_per_lanaguage),
        "top_card_data_pipeline": [match_obj] + copy.deepcopy(top_card_data_pipeline),
        "side_card_data_pipeline": [match_obj] + copy.deepcopy(side_card_data_pipeline),
        "area_chart_entities_pipeline": [match_obj] + copy.deepcopy(area_chart_entities_pipeline),
        "entities_classes_vs_emotion_pipeline": [match_obj] + copy.deepcopy(entities_classes_vs_emotion_pipeline),
        "applications_vs_topics_pipeline": [match_obj] + copy.deepcopy(applications_vs_topics_pipeline),
        "topics_vs_emotions_pipeline": [match_obj] + copy.deepcopy(topics_vs_emotions_pipeline),
        "entities_vs_emotions_pipeline": [match_obj] + copy.deepcopy(entities_vs_emotions_pipeline),
        "entities_vs_applications_pipeline": [match_obj] + copy.deepcopy(entities_vs_applications_pipeline),
        "entities_vs_topics_pipeline": [match_obj] + copy.deepcopy(entities_vs_topics_pipeline)
    }