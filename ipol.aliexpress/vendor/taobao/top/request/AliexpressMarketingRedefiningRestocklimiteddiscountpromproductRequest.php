<?php
/**
 * TOP API: aliexpress.marketing.redefining.restocklimiteddiscountpromproduct request
 * 
 * @author auto create
 * @since 1.0, 2019.05.30
 */
class AliexpressMarketingRedefiningRestocklimiteddiscountpromproductRequest
{
	/** 
	 * 入参
	 **/
	private $ldDTO;
	
	private $apiParas = array();
	
	public function setLdDTO($ldDTO)
	{
		$this->ldDTO = $ldDTO;
		$this->apiParas["ld_d_t_o"] = $ldDTO;
	}

	public function getLdDTO()
	{
		return $this->ldDTO;
	}

	public function getApiMethodName()
	{
		return "aliexpress.marketing.redefining.restocklimiteddiscountpromproduct";
	}
	
	public function getApiParas()
	{
		return $this->apiParas;
	}
	
	public function check()
	{
		
	}
	
	public function putOtherTextParam($key, $value) {
		$this->apiParas[$key] = $value;
		$this->$key = $value;
	}
}
