<?php

/**
 * 商品分国家定价规则数据，建议使用新格式，请参考：https://developers.aliexpress.com/doc.htm?docId=109575&docType=1
 * @author auto create
 */
class AeopNationalQuoteConfiguration
{
	
	/** 
	 * jsonArray格式的分国家定价规则数据。 1)基于基准价格按比例配置的数据格式：[{"shiptoCountry":"US","percentage":"5"},{"shiptoCountry":"RU","percentage":"-2"}] 其中shiptoCountry：ISO两位的国家编码（目前支持11个国家：RU,US,CA,ES,FR,UK,NL,IL,BR,CL,AU）， percentage：相对于基准价的调价比例（百分比整数，支持负数，当前限制>=-30 && <=100）；如果需要删除分国家报价则将该值设为空串""
	 **/
	public $configuration_data;
	
	/** 
	 * 分国家定价规则类型[percentage：基于基准价格按比例配置]
	 **/
	public $configuration_type;	
}
?>