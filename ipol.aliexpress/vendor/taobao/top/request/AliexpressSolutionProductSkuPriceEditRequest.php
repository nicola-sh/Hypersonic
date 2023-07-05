<?php
/**
 * TOP API: aliexpress.solution.product.sku.price.edit request
 * 
 * @author auto create
 * @since 1.0, 2019.08.13
 */
class AliexpressSolutionProductSkuPriceEditRequest
{
	/** 
	 * input parameters
	 **/
	private $editProductSkuPriceRequest;
	
	private $apiParas = array();
	
	public function setEditProductSkuPriceRequest($editProductSkuPriceRequest)
	{
		$this->editProductSkuPriceRequest = $editProductSkuPriceRequest;
		$this->apiParas["edit_product_sku_price_request"] = $editProductSkuPriceRequest;
	}

	public function getEditProductSkuPriceRequest()
	{
		return $this->editProductSkuPriceRequest;
	}

	public function getApiMethodName()
	{
		return "aliexpress.solution.product.sku.price.edit";
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
