<?php

/**
 * input parameters
 * @author auto create
 */
class EditItemSkuPriceDto
{
	
	/** 
	 * discount price, which should be always cheaper than the value of price. The field "price" and "discount_price" could not be both empty.
	 **/
	public $discount_price;
	
	/** 
	 * price of the sku, which should always more expensive that the value of discount_price. The field "price" and "discount_price" could not be both empty.
	 **/
	public $price;
	
	/** 
	 * aliexpress product id
	 **/
	public $product_id;
	
	/** 
	 * sku_code, of which the price will be updated
	 **/
	public $sku_code;	
}
?>