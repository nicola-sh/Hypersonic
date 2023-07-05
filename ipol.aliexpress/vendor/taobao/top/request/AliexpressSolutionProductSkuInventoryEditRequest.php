<?php
/**
 * TOP API: aliexpress.solution.product.sku.inventory.edit request
 * 
 * @author auto create
 * @since 1.0, 2019.08.13
 */
class AliexpressSolutionProductSkuInventoryEditRequest
{
	/** 
	 * input parameters
	 **/
	private $editProductSkuInventoryRequest;
	
	private $apiParas = array();
	
	public function setEditProductSkuInventoryRequest($editProductSkuInventoryRequest)
	{
		$this->editProductSkuInventoryRequest = $editProductSkuInventoryRequest;
		$this->apiParas["edit_product_sku_inventory_request"] = $editProductSkuInventoryRequest;
	}

	public function getEditProductSkuInventoryRequest()
	{
		return $this->editProductSkuInventoryRequest;
	}

	public function getApiMethodName()
	{
		return "aliexpress.solution.product.sku.inventory.edit";
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
