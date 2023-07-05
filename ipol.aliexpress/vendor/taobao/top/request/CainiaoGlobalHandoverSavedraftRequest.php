<?php
/**
 * TOP API: cainiao.global.handover.savedraft request
 * 
 * @author auto create
 * @since 1.0, 2020.01.16
 */
class CainiaoGlobalHandoverSavedraftRequest
{
	/** 
	 * ISV英文名称
	 **/
	private $client;
	
	/** 
	 * 多语言
	 **/
	private $locale;
	
	/** 
	 * 需要组装大包的小包编码集合，最多限制200个小包
	 **/
	private $orderCodeList;
	
	/** 
	 * 揽收信息，默认为关联小包上的揽收地址信息
	 **/
	private $pickupInfo;
	
	/** 
	 * 备注
	 **/
	private $remark;
	
	/** 
	 * 退件信息，默认为关联小包上的退件信息
	 **/
	private $returnInfo;
	
	/** 
	 * 大包重量
	 **/
	private $weight;
	
	/** 
	 * 大包重量单位，默认g
	 **/
	private $weightUnit;
	
	private $apiParas = array();
	
	public function setClient($client)
	{
		$this->client = $client;
		$this->apiParas["client"] = $client;
	}

	public function getClient()
	{
		return $this->client;
	}

	public function setLocale($locale)
	{
		$this->locale = $locale;
		$this->apiParas["locale"] = $locale;
	}

	public function getLocale()
	{
		return $this->locale;
	}

	public function setOrderCodeList($orderCodeList)
	{
		$this->orderCodeList = $orderCodeList;
		$this->apiParas["order_code_list"] = $orderCodeList;
	}

	public function getOrderCodeList()
	{
		return $this->orderCodeList;
	}

	public function setPickupInfo($pickupInfo)
	{
		$this->pickupInfo = $pickupInfo;
		$this->apiParas["pickup_info"] = $pickupInfo;
	}

	public function getPickupInfo()
	{
		return $this->pickupInfo;
	}

	public function setRemark($remark)
	{
		$this->remark = $remark;
		$this->apiParas["remark"] = $remark;
	}

	public function getRemark()
	{
		return $this->remark;
	}

	public function setReturnInfo($returnInfo)
	{
		$this->returnInfo = $returnInfo;
		$this->apiParas["return_info"] = $returnInfo;
	}

	public function getReturnInfo()
	{
		return $this->returnInfo;
	}

	public function setWeight($weight)
	{
		$this->weight = $weight;
		$this->apiParas["weight"] = $weight;
	}

	public function getWeight()
	{
		return $this->weight;
	}

	public function setWeightUnit($weightUnit)
	{
		$this->weightUnit = $weightUnit;
		$this->apiParas["weight_unit"] = $weightUnit;
	}

	public function getWeightUnit()
	{
		return $this->weightUnit;
	}

	public function getApiMethodName()
	{
		return "cainiao.global.handover.savedraft";
	}
	
	public function getApiParas()
	{
		return $this->apiParas;
	}
	
	public function check()
	{
		
		RequestCheckUtil::checkNotNull($this->client,"client");
		RequestCheckUtil::checkNotNull($this->orderCodeList,"orderCodeList");
		RequestCheckUtil::checkMaxListSize($this->orderCodeList,200,"orderCodeList");
	}
	
	public function putOtherTextParam($key, $value) {
		$this->apiParas[$key] = $value;
		$this->$key = $value;
	}
}
